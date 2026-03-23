# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "google-genai>=1.67.0",
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
from google import genai
from google.genai import types

from prompt_asset.loader import PromptTemplateCollections
from prompt_asset.prompt_cache import PromptCacheSystem, get_cache_id

load_dotenv() # load gemini API key

def read_benchmark(json_dir: str):
    json_files = glob.glob(os.path.join(json_dir, "**", "*.json"), recursive=True)

    video_names = set()
    benchmarks = []

    for json_file in tqdm(json_files, desc="Scanning JSON files"):
        _affirmative_questions = []
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            benchmarks.append(data)

        except Exception as e:
            tqdm.write(f"Failed to read {json_file}: {e}")

    return benchmarks

def call_llm(client, model_id: str, prompt: str, vid: str) -> List[Dict]:
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
        breakpoint()
        
        raw_text = response.text.strip()
        
        # Fallback cleanup just in case
        raw_text = response.text.strip()
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
        print("Sometimes, if life is too harsh for you and the atomic facts cannot be created, please increase `max_output_tokens`!")
        evaluation_dataset = []

    return evaluation_dataset

if __name__ == "__main__":
    json_dir = sys.argv[1] if len(sys.argv) > 1 else "benchmark"
    # recompute = sys.argv[2] if len(sys.argv) > 2 else "No"
    benchmarks = read_benchmark(json_dir)
    aug_query_prompt = PromptTemplateCollections['augment_question'].content
    atomic_facts_storage = PromptCacheSystem("cache/atomic_facts")
    cache_sys = PromptCacheSystem("cache/sub_questions")

    key_to_use = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=key_to_use)
    model = "gemini-3-flash-preview"

    # batch size
    batch_size = 5

    # get a subset
    benchmarks = benchmarks[:5]
    for sample in tqdm(benchmarks):
        
        sub_questions = []
        cache_id = get_cache_id(sample['video_name'])
        if cache_sys.exist(cache_id):
            continue
        # get atomic facts
        if not atomic_facts_storage.exist(cache_id):
            tqdm.write(f"Error. Not atomic facts available for {cache_id}")
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
            sub_questions.extend(call_llm(client, model, final_prompt, cache_id))
            
        
        sample['sub-questions'] = sub_questions
        cache_sys.push(cache_id, sample)