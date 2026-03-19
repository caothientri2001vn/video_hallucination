# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "google-genai>=1.67.0",
#     "python-dotenv>=1.2.2",
#     "tqdm>=4.67.3"
# ]
# ///
"""
Auto-generate candidate sub-questions for consistency evaluation.
Uses the Gemini API to decompose target questions into atomic predicates,
then converts each predicate into a yes/no sub-question.

Output: JSON ready for annotator verification.
"""

import os
import json
import random
from dotenv import load_dotenv
from google import genai
from google.genai import types
from dataclasses import dataclass
from typing import List, Any, Tuple, Dict, Optional
from tqdm import tqdm

@dataclass
class PredicateType:
    predicate_type: str
    predicate_definition: str

class PromptCacheSystem:
    def __init__(self, cache_dir: str = "prompt_library"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def push(self, sample_idx: str, prompt: str, response: List[Any]):
        cache_path = os.path.join(self.cache_dir, f"sample_{sample_idx}.json")
        with open(cache_path, 'w+', encoding='utf-8') as fout:
            json.dump({"prompt": prompt, "response": response}, fout)

    def get(self, sample_idx: str) -> Optional[List[str]]:
        if not self.exist(sample_idx):
            return None
        cache_path = os.path.join(self.cache_dir, f"sample_{sample_idx}.json")
        with open(cache_path, encoding='utf-8') as fin:
            data = json.load(fin)

        return data

    def exist(self, sample_idx: str) -> bool:
        cache_path = os.path.join(self.cache_dir, f"sample_{sample_idx}.json")
        return os.path.exists(cache_path)

# Load environment variables from the .env file
load_dotenv()

SINGLE_PREDICATE_DECOMPOSITION_PROMPT = """You are helping build a video understanding benchmark.

Given a yes/no question about a video and its ground-truth answer, decompose the question
into atomic facts that are necessary for the target answer, but ONLY for the predicate type:
{predicate_type}

Your job:
- Identify 1-4 necessary atomic facts that belong strictly to the predicate type {predicate_type}.
- Convert each atomic fact into an independent yes/no sub-question.
- Every sub-question must be answerable from the video alone.
- Every sub-question must have a definite expected answer.
- Every sub-question must be logically necessary for the original target answer.

Important constraints:
1. Use ONLY predicate type: {predicate_type}
2. Do NOT mix in other predicate types.
3. If this target question cannot be meaningfully decomposed into necessary sub-questions of type {predicate_type},
   return an empty decomposition.
4. If the target answer is "No", decompose the facts that would need to hold for the answer to be "Yes".
   Some expected answers may therefore be "No".
5. Prefer fewer, high-quality sub-questions over many weak ones.
6. Do NOT generate trivial or generic sub-questions.

Predicate type definition:
{predicate_definition}

Output ONLY valid JSON matching this structure:
{{
  "predicate_type": "{predicate_type}",
  "decomposition": [
    {{
      "atomic_fact": "description of the fact being tested",
      "sub_question": "the yes/no question",
      "expected_answer": "Yes|No",
      "reasoning": "why this fact is necessary for the target answer"
    }}
  ]
}}

Target question: {question}
Ground-truth answer: {answer}
Video context (if available): {context}
"""

identity_predicate_type = PredicateType(
    predicate_type="IDENTITY",
    predicate_definition=(
        "IDENTITY checks who or what an entity is, or whether an entity at one moment "
        "is the same as an entity at another moment. Use this only for entity identity, "
        "coreference, re-identification, role assignment, or object/person matching across time."
    ),
)

state_predicate_type = PredicateType(
    predicate_type="STATE",
    predicate_definition=(
        "STATE checks an attribute, condition, pose, location, status, or property of an entity "
        "at a relevant moment or across moments. Use this only for facts like open/closed, "
        "standing/sitting, holding/not holding, on/off, inside/outside, or changed state."
    ),
)

relation_predicate_type = PredicateType(
    predicate_type="RELATION",
    predicate_definition=(
        "RELATION checks how two or more entities are connected in space, action, contact, "
        "ownership, interaction, or other dependency. Use this only for facts like next to, "
        "holding, touching, following, facing, giving to, or interacting with."
    ),
)

temporal_predicate_type = PredicateType(
    predicate_type="TEMPORAL",
    predicate_definition=(
        "TEMPORAL checks ordering, before/after relations, duration-sensitive facts, sequence, "
        "or change over time. Use this only for facts involving event order, whether something "
        "happened first/last, before/after another event, or whether a change occurred over time."
    ),
)

count_predicate_type = PredicateType(
    predicate_type="COUNT",
    predicate_definition=(
        "COUNT checks how many entities, events, or occurrences are present. Use this only for "
        "facts involving exact number, plurality, repetition, or numerical comparison."
    ),
)

existence_predicate_type = PredicateType(
    predicate_type="EXISTENCE",
    predicate_definition=(
        "EXISTENCE checks whether an entity, event, action, or situation appears or occurs in the video. "
        "Use this only for presence/absence questions, not for identity, state, relation, order, or count."
    ),
)

PREDICATE_TYPES: List[PredicateType] = [
    identity_predicate_type,
    state_predicate_type,
    relation_predicate_type,
    temporal_predicate_type,
    count_predicate_type,
    existence_predicate_type
]

cache_sys = PromptCacheSystem()

def call_llm(client: "API", model_id: str, cache_sys: PromptCacheSystem, query: Dict, predicate_type: PredicateType, counter: int, recompute: bool = False) -> Dict:
    qid = query['question_id']
    cache_id = f"{qid}_{predicate_type.predicate_type}"
    if cache_sys.exist(cache_id) and not recompute:
        output = cache_sys.get(cache_id)
        subs = output['response']
    else:
        prompt = SINGLE_PREDICATE_DECOMPOSITION_PROMPT.format(
            predicate_type=predicate_type.predicate_type,
            predicate_definition=predicate_type.predicate_definition,
            question=query["question"],
            answer=query["answer"],
            context=query.get("video_title", "No additional context"),
        )
        try:
            # Call Gemini API with JSON enforcement using the new SDK syntax
            response = client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=8000,
                    response_mime_type="application/json",
                )
            )
            
            text = response.text.strip()
            
            # Fallback cleanup just in case
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                    
            parsed = json.loads(text)
            subs = parsed.get("decomposition", [])
            
        except Exception as e:
            print(f"  [{counter}] Error for Q{query.get('question_id', 'Unknown')} with predicate type: {predicate_type.predicate_type}: {e}")
            subs = []
        if len(subs) > 0:
            cache_sys.push(cache_id, prompt, subs)
    return {"subs": subs, "predicate_type": predicate_type.predicate_type}


def generate_sub_questions(
    questions: list[dict],
    model: str = "gemini-2.5-pro", # gemini-3-flash-preview | gemini-2.5-pro
    api_key: str = None,
) -> list[dict]:
    # Retrieve the API key from the environment (loaded from .env)
    key_to_use = api_key or os.getenv("GEMINI_API_KEY")
    if not key_to_use:
        raise ValueError("GEMINI_API_KEY not found. Please ensure it is set in your .env file.")

    # Initialize the new google-genai client
    client = genai.Client(api_key=key_to_use)
    
    results = []

    for i, q in enumerate(tqdm(questions)):
        for pred_type in PREDICATE_TYPES:
            output = call_llm(
                client,
                model,
                cache_sys,
                q,
                pred_type,
                counter=i+1
            )
            subs = output['subs']
            results.append({
                **q,
                "candidate_sub_questions": [
                    {
                        "sub_question": s["sub_question"],
                        "expected_answer": s["expected_answer"],
                        "predicate_type": pred_type.predicate_type,
                        "atomic_fact": s["atomic_fact"],
                        "reasoning": s.get("reasoning", ""),
                        "annotator_verdict": None,
                        "annotator_corrected_answer": None,
                        "annotator_corrected_text": None,
                    }
                    for s in subs
                ],
            })
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(questions)}")

    return results

def build_annotation_file(generated: list[dict], output_path: str):
    """
    Create annotation file for human reviewers.
    Each sub-question gets a simple accept/reject/edit interface.
    """
    annotation_tasks = []
    sub_id_counter = 10000

    for item in generated:
        for sub in item["candidate_sub_questions"]:
            sub_id_counter += 1
            annotation_tasks.append({
                "task_id": sub_id_counter,
                "video_path": item.get("video_path", ""),
                "video_title": item.get("video_title", ""),
                # Target question context
                "target_question_id": item["question_id"],
                "target_question": item["question"],
                "target_answer": item["answer"],
                # Sub-question to verify
                "sub_question": sub["sub_question"],
                "proposed_answer": sub["expected_answer"],
                "predicate_type": sub["predicate_type"],
                "atomic_fact": sub["atomic_fact"],
                # Annotator fields
                "instructions": (
                    "Watch the video. Then answer these 3 questions:\n"
                    "1. Is the proposed answer correct? (yes/no/ambiguous)\n"
                    "2. Is this sub-question logically necessary for the "
                    "target question? (yes/no)\n"
                    "3. Is the sub-question clear and unambiguous? "
                    "(yes / no — if no, provide edited version)"
                ),
                "answer_correct": None,        # "yes" | "no" | "ambiguous"
                "logically_necessary": None,    # "yes" | "no"
                "clear_and_unambiguous": None,  # "yes" | "no"
                "edited_question": None,        # str if edited
                "corrected_answer": None,       # str if answer was wrong
                "annotator_id": None,
                "annotator_notes": None,
            })

    with open(output_path, "w") as f:
        json.dump(annotation_tasks, f, indent=2)
    print(f"Annotation file saved: {output_path}")
    print(f"  {len(annotation_tasks)} sub-questions to verify")
    print(f"  from {len(generated)} target questions")
    return annotation_tasks


def compile_final_benchmark(
    annotation_path: str,
    original_questions: list[dict],
    output_path: str,
):
    """
    After annotation, compile the final benchmark with verified groups.
    Keeps only sub-questions where:
      - answer_correct == "yes"
      - logically_necessary == "yes"  
      - clear_and_unambiguous == "yes"
    """
    with open(annotation_path) as f:
        annotations = json.load(f)

    # Filter accepted sub-questions
    accepted = [
        a for a in annotations
        if a.get("answer_correct") == "yes"
        and a.get("logically_necessary") == "yes"
        and a.get("clear_and_unambiguous") == "yes"
    ]

    # Group by target question
    target_to_subs = {}
    for a in accepted:
        tid = a["target_question_id"]
        if tid not in target_to_subs:
            target_to_subs[tid] = []
        target_to_subs[tid].append({
            "question": a.get("edited_question") or a["sub_question"],
            "answer": a.get("corrected_answer") or a["proposed_answer"],
            "predicate_type": a["predicate_type"],
        })

    # Build final benchmark
    final = []
    qid_counter = 0
    for orig in original_questions:
        qid_counter += 1
        target_entry = {
            **orig,
            "question_id": qid_counter,
            "target_question_id": None,  # this IS the target
            "sub_question_ids": [],
        }
        final.append(target_entry)

        subs = target_to_subs.get(orig["question_id"], [])
        for sub in subs:
            qid_counter += 1
            final.append({
                "video_path": orig.get("video_path", ""),
                "question_id": qid_counter,
                "question": sub["question"],
                "answer": sub["answer"],
                "group_id": target_entry["question_id"],
                "target_question_id": target_entry["question_id"],
                "predicate_type": sub["predicate_type"],
            })
            target_entry["sub_question_ids"].append(qid_counter)

    with open(output_path, "w") as f:
        json.dump(final, f, indent=2)

    groups_with_subs = sum(1 for q in final
                           if q["target_question_id"] is None
                           and len(q.get("sub_question_ids", [])) > 0)
    total_subs = sum(1 for q in final if q["target_question_id"] is not None)

    print(f"Final benchmark saved: {output_path}")
    print(f"  {len(final)} total questions")
    print(f"  {groups_with_subs} groups with sub-questions")
    print(f"  {total_subs} sub-questions total")
    print(f"  Avg {total_subs/max(groups_with_subs,1):.1f} subs per group")


# ---- Example usage ----
if __name__ == "__main__":
    # sample_questions = [
    #     {
    #         "question_id": 1,
    #         "question": "Is the woman who walks toward us at the beginning the one who later stands up and leaves?",
    #         "answer": "No",
    #         "video_path": "videos/0045.mp4",
    #         "video_title": "Crazy Rich Asians - Mahjong Scene",
    #     },
    #     {
    #         "question_id": 3,
    #         "question": "Did the man on the left open three boxes?",
    #         "answer": "No",
    #         "video_path": "videos/0046.mp4",
    #         "video_title": "Box of Lies with Chris Pratt",
    #     }
    # ]
    with open("reproducible_sample_questions.json", encoding='utf-8') as fin:
        sample_questions = json.load(fin)
    # sample_questions = sample_questions[:10]

    # Make sure you have the GEMINI_API_KEY environment variable set!
    # export GEMINI_API_KEY="your-api-key-here"
    
    # Step 1: Auto-generate candidates
    # generated = generate_sub_questions(sample_questions, model = "gemini-3-flash-preview")
    # generated = generate_sub_questions(sample_questions, model = "gemini-2.5-pro")
    generated = generate_sub_questions(sample_questions, model = "gemini-3-pro-preview")
    
    # Step 2: Create annotation file
    build_annotation_file(generated, "annotation_tasks.json")
    
    # Step 3: (Annotators fill in the file)
    
    # Step 4: Compile final benchmark
    # compile_final_benchmark("annotation_tasks_completed.json",
    #                         sample_questions, "benchmark_final.json")