#!/bin/bash
#SBATCH --job-name=random_dolly_train_gemma_3_4b
#SBATCH --partition=AI4Good_P
#SBATCH --gres=gpu:1

python ../train_gemma.py --data_path /mnt/petrelfs/luzhenghao/safe_useful/random_results/dolly-clean_random_1000.json --output_path mnt/petrelfs/luzhenghao/safe_useful/gemma_3_4b_output/gemma_3_4b_bi_random_1000_dolly