import os
import json
import random

def get_random_json_files(directory_path: str, num_files: int = 5, seed: int = None) -> list[str]:
    """
    Scans the directory for JSON files and randomly selects a specified number.
    """
    if seed is not None:
        random.seed(seed)
        
    # Find all files in the directory ending with .json
    all_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
    
    # If there are fewer files than requested, just return what is available
    if len(all_files) <= num_files:
        print(f"Found {len(all_files)} files, which is less than or equal to requested {num_files}.")
        selected_files = all_files
    else:
        selected_files = random.sample(all_files, num_files)
        
    # Return full paths
    return [os.path.join(directory_path, f) for f in selected_files]

def extract_and_format_questions(file_paths: list[str], max_instances: int = 10, seed: int = None) -> list[dict]:
    """
    Extracts the video name, questions, and answers from the files, 
    formats them to match the target schema, and limits the output.
    """
    if seed is not None:
        random.seed(seed)
        
    all_extracted_instances = []
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                video_name = data.get('video_name', 'unknown_video.mp4')
                questions = data.get('questions', [])
                answers = data.get('answers', [])
                
                # Pair up questions and answers
                for q, a in zip(questions, answers):
                    all_extracted_instances.append({
                        "question": q,
                        "answer": a,
                        "video_path": f"raw_data/{video_name}", # Matching previous structure
                    })
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error reading {file_path}: {e}")
            
    # Shuffle the combined list so we get a good mix of questions across the 5 files
    # random.shuffle(all_extracted_instances)
    
    # Limit to the requested number of instances (10)
    selected_instances = all_extracted_instances[:max_instances]
    
    # Add sequential question_ids to the final selection
    for index, instance in enumerate(selected_instances):
        # Insert question_id at the beginning of the dictionary
        final_dict = {"question_id": index + 1}
        final_dict.update(instance)
        selected_instances[index] = final_dict
        
    return selected_instances

def save_to_json(data: list[dict], output_file_path: str):
    """
    Saves the list of dictionaries into a formatted JSON file.
    """
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Successfully saved {len(data)} instances to '{output_file_path}'.")

def main(directory_path: str, output_path: str):
    """
    Main orchestration function.
    """
    # Using a fixed seed (e.g., 42) guarantees the same random selections every time.
    # Remove the seed parameter if you want true randomness on every run.
    RANDOM_SEED = 42
    
    # 1 & 2: Get directory and randomly select 5 files
    selected_files = get_random_json_files(directory_path, num_files=100, seed=RANDOM_SEED)
    print("Selected files for processing:")
    for f in selected_files:
        print(f" - {f}")
        
    # 3 & 4: Extract data, format it, and limit to 10 instances
    sample_questions = extract_and_format_questions(selected_files, max_instances=1500, seed=RANDOM_SEED)
    
    # 5: Store into a JSON file
    save_to_json(sample_questions, output_path)

if __name__ == "__main__":
    # --- Instructions to run ---
    # 1. Update TARGET_DIRECTORY to the folder containing your JSON files.
    # 2. Update OUTPUT_FILE to your desired save location.
    
    TARGET_DIRECTORY = "benchmark" 
    OUTPUT_FILE = "reproducible_sample_questions.json"
    
    # Ensure the directory exists before running to avoid errors
    if os.path.exists(TARGET_DIRECTORY):
        main(TARGET_DIRECTORY, OUTPUT_FILE)
    else:
        print(f"Please create the directory '{TARGET_DIRECTORY}' and add some JSON files to test.")