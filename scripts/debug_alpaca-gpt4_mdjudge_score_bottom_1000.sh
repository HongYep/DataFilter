#!/bin/bash
#SBATCH --job-name=alpaca-gpt4_mdjudge_score_bottom_1000
#SBATCH --partition=AI4Good_P
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/petrelfs/luzhenghao/safe_useful/gemma_2_9b_output/alpaca-gpt4_mdjudge_score_bottom_1000.log
python /mnt/petrelfs/luzhenghao/safe_useful/train_gemma_2.py --data_path /mnt/petrelfs/luzhenghao/safe_useful/mdjudge_score_results/alpaca-gpt4_mdjudge_score_bottom_1000.json --output_path /mnt/petrelfs/luzhenghao/safe_useful/gemma_2_9b_output/alpaca-gpt4_mdjudge_score_bottom_1000
