MODEL_NAME="/mnt/petrelfs/lihao1/trustai/share/models/meta-llama/Llama-Guard-3-8B"
srun -p AI4Good_L1 --gres=gpu:1 -J eval vllm serve $MODEL_NAME --dtype auto --api-key token-pjlab --port 8612