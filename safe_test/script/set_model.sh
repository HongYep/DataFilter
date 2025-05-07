MODEL_NAME="/mnt/petrelfs/share_data/safety_verifier/models/gemma-3-4b-it"
# /mnt/petrelfs/share_data/safety_verifier/models/gemma-3-4b-it
# /mnt/petrelfs/share_data/safety_verifier/models/Llama-3.1-8B-Instruct
# /mnt/petrelfs/share_data/safety_verifier/models/Qwen2.5-7B-Instruct
srun -p AI4Good_P --gres=gpu:1 -J eval vllm serve $MODEL_NAME --dtype auto --api-key token-pjlab --port 8631