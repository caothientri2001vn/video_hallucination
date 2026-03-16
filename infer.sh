# CUDA_VISIBLE_DEVICES=1 python vibe_check_video_qa.py \
#     --model-id weights/qwen3_vl_8b_inst \
#     --video-path raw_data/0026_09uVWWLKNdc.mp4 \
#     --question "Did the yellow ball stay at the starting point during the first experiment?"

# CUDA_VISIBLE_DEVICES=1 python generate_result.py --model_id weights/qwen3_vl_8b_inst --mode sample_2.json

# CUDA_VISIBLE_DEVICES=0 python benchmark.py \
#     --model_id weights/qwen3_vl_8b_inst

CUDA_VISIBLE_DEVICES=0 python benchmark.py \
    --model_id weights/qwen3_vl_8b_think \
    --cache_dir cache_think \
    --max_new_tokens 1024