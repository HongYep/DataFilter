#!/bin/bash
#SBATCH --job-name=magicoder_moderation_score_top_1000
#SBATCH --partition=AI4Good_P
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_12b_output/magicoder_moderation_score_top_1000.log
python /mnt/petrelfs/luzhenghao/safe_useful/train_gemma_2.py --data_path /mnt/petrelfs/luzhenghao/safe_useful/moderation_score_results/magicoder_moderation_score_top_1000.json --output_path /mnt/petrelfs/luzhenghao/safe_useful/gemma_3_12b_output/magicoder_moderation_score_top_1000
