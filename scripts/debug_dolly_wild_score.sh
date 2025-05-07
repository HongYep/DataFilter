#!/bin/bash
#SBATCH --job-name=dolly_wild_score_top_1000
#SBATCH --partition=AI4Good_L
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/petrelfs/luzhenghao/safe_useful/llama3_output/dolly_wild_score_top_1000.log
python /mnt/petrelfs/luzhenghao/safe_useful/train_llama3.py --data_path /mnt/petrelfs/luzhenghao/safe_useful/wild_score_results/dolly_wild_score_top_1000.json --output_path /mnt/petrelfs/luzhenghao/safe_useful/llama3_output/dolly_wild_score_top_1000
