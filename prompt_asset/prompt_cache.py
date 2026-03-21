from typing import List, Any, Optional, Dict
import os
import json

def get_cache_id(video_name: str):
    return video_name.split(".")[0] # exclude the extension

class PromptCacheSystem:
    def __init__(self, cache_dir: str = "prompt_library"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def push(self, sample_idx: str, obj: Dict):
        cache_path = os.path.join(self.cache_dir, f"sample_{sample_idx}.json")
        with open(cache_path, 'w+', encoding='utf-8') as fout:
            json.dump(obj, fout)

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