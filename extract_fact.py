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
from typing import List

from dotenv import load_dotenv
from google import genai
from google.genai import types

from prompt_asset.loader import PromptTemplateCollections
from prompt_asset.prompt_cache import PromptCacheSystem, get_cache_id

load_dotenv() # load gemini API key

def extract_affirmative_question(json_dir: str):
    json_files = glob.glob(os.path.join(json_dir, "**", "*.json"), recursive=True)

    total_questions = 0
    video_names = set()
    total_yes = 0
    total_no = 0
    total_other = 0
    affirmative_questions = []

    for json_file in tqdm(json_files, desc="Scanning JSON files"):
        _affirmative_questions = []
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            total_questions += len(data.get("questions", []))
            video_name = data.get("video_name", "")
            if video_name:
                video_names.add(video_name)

            questions = data.get("questions", [])
            for idx, answer in enumerate(data.get("answers", [])):
                normalized = str(answer).strip().lower()
                if normalized == "yes":
                    total_yes += 1
                    _affirmative_questions.append(questions[idx])
                elif normalized == "no":
                    total_no += 1
                else:
                    total_other += 1
                    tqdm.write(f"Unexpected answer '{answer}' in {json_file}")

        except Exception as e:
            tqdm.write(f"Failed to read {json_file}: {e}")
        
        affirmative_questions.append({"video_name": video_name, "a_questions": _affirmative_questions})

    return affirmative_questions

def call_llm(client, model_id: str, prompt: str, vid: str) -> List[str]:
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
        
        raw_text = response.text.strip()
        
        # Fallback cleanup just in case
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
        raw_text = raw_text.strip()
                
        unit_facts_list = json.loads(raw_text)
        
    except Exception as e:
        print(f"This is not good! Please check {vid} for more information")
        print(f"The error is {e}")
        print("Sometimes, if life is too harsh for you and the atomic facts cannot be created, please increase `max_output_tokens`!")
        unit_facts_list = []

    return unit_facts_list

if __name__ == "__main__":
    json_dir = sys.argv[1] if len(sys.argv) > 1 else "benchmark"
    # recompute = sys.argv[2] if len(sys.argv) > 2 else "No"
    affirmative_questions = extract_affirmative_question(json_dir)
    extract_fact_prompt = PromptTemplateCollections['extract_fact'].content
    cache_sys = PromptCacheSystem("cache/atomic_facts")

    key_to_use = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=key_to_use)
    model = "gemini-3-flash-preview"

    for affirmative_question in tqdm(affirmative_questions):
        cache_id = get_cache_id(affirmative_question['video_name'])
        # if cache_sys.exist(cache_id) and recompute == "No":
        if cache_sys.exist(cache_id):
            continue

        _a_queries = "\n".join(affirmative_question['a_questions'])
        final_prompt = extract_fact_prompt.replace("[INSERT_QUESTIONS_HERE]", _a_queries)
        # breakpoint()
        atomic_facts = call_llm(client, model, final_prompt, cache_id)
        cache_sys.push(cache_id, {
            "video_name": affirmative_question['video_name'],
            "affirmative_questions": affirmative_question['a_questions'],
            "prompt": final_prompt,
            "atomic_facts": atomic_facts
        })
        