MODEL_NAME=/mnt/petrelfs/lihao1/trustai/share/models/allenai/wildguard
srun -p AI4Good_S1 --gres=gpu:1 -J eval vllm serve $MODEL_NAME --dtype auto --api-key token-pjlab --port 8612