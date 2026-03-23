# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "anthropic>=0.40.0",
#     "python-dotenv>=1.2.2",
#     "tqdm>=4.67.3"
# ]
# ///

import os
import sys
import glob
import json
from tqdm import tqdm
from typing import List, Dict

from dotenv import load_dotenv
import anthropic

from prompt_asset.loader import PromptTemplateCollections
from prompt_asset.prompt_cache import PromptCacheSystem, get_cache_id

load_dotenv()  # load Anthropic API key

def read_benchmark(json_dir: str):
    json_files = glob.glob(os.path.join(json_dir, "**", "*.json"), recursive=True)

    benchmarks = []

    for json_file in tqdm(json_files, desc="Scanning JSON files"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            benchmarks.append(data)

        except Exception as e:
            tqdm.write(f"Failed to read {json_file}: {e}")

    return benchmarks

def call_llm(client: anthropic.Anthropic, model_id: str, prompt: str, vid: str) -> List[Dict]:
    try:
        response = client.messages.create(
            model=model_id,
            max_tokens=8000,
            temperature=0.3,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            system="You are a helpful assistant. Always respond with valid JSON only, with no additional text, explanation, or markdown formatting."
        )

        raw_text = response.content[0].text.strip()

        # Fallback cleanup just in case
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
        raw_text = raw_text.strip()

        evaluation_dataset = json.loads(raw_text)

    except Exception as e:
        print(f"This is not good! Please check {vid} for more information")
        print(f"The error is {e}")
        print("Sometimes, if life is too harsh for you and the atomic facts cannot be created, please increase `max_tokens`!")
        evaluation_dataset = []

    return evaluation_dataset

def fake_call_llm(client, model_id, prompt, vid) -> List[Dict]:
    with open(f"cache_{vid}.txt", 'w+') as fout:
        fout.write(prompt)
    return ['1']

if __name__ == "__main__":
    json_dir = sys.argv[1] if len(sys.argv) > 1 else "benchmark"
    benchmarks = read_benchmark(json_dir)
    aug_query_prompt = PromptTemplateCollections['augment_question'].content
    atomic_facts_storage = PromptCacheSystem("cache/atomic_facts")
    cache_sys = PromptCacheSystem("cache/sub_questions_claude")

    key_to_use = os.getenv("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=key_to_use)
    model = "claude-sonnet-4-5"

    # batch size
    batch_size = 5

    # get a subset
    benchmarks = benchmarks[:6]
    for sample in tqdm(benchmarks):
        sub_questions = []
        cache_id = get_cache_id(sample['video_name'])
        if cache_sys.exist(cache_id):
            continue

        # get atomic facts
        if not atomic_facts_storage.exist(cache_id):
            tqdm.write(f"Error. No atomic facts available for {cache_id}")
            raise Exception("Abort mission")    
        atomic_facts = atomic_facts_storage.get(cache_id)['atomic_facts']
        facts_string = "\n".join(f"- {fact}" for fact in atomic_facts)
        original_qa_data = [{"question": q, "answer": a} for q, a in zip(sample['questions'], sample['answers'])]

        for i in range(0, len(original_qa_data), batch_size):
            batch_ori_qa_data = original_qa_data[i : i + batch_size]
            qa_string = "\n".join(
                f"- Question: {item['question']} | Answer: {item['answer']}"
                for item in batch_ori_qa_data
            )
            final_prompt = aug_query_prompt.replace("[INSERT_UNIT_FACTS_HERE]", facts_string)
            final_prompt = final_prompt.replace("[INSERT_QA_PAIRS_HERE]", qa_string)
            _sub_questions = call_llm(client, model, final_prompt, cache_id)
            sub_questions.extend(_sub_questions)

        sample['sub-questions'] = sub_questions
        sample['atomic-facts'] = atomic_facts
        cache_sys.push(cache_id, sample)