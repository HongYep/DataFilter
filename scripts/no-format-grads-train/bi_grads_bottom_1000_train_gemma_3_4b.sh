#!/bin/bash
#SBATCH --job-name=random_cleaned_train_gemma_3_4b
#SBATCH --partition=AI4Good_P
#SBATCH --gres=gpu:1

python ../../train_gemma.py --data_path /mnt/petrelfs/luzhenghao/safe_useful/grads_sort_results/alpaca_no_format_grads_bi_bottom_1000.json --output_path /mnt/petrelfs/luzhenghao/safe_useful/gemma_3_4b_output/alpaca_no_format_grads_bi_bottom_1000