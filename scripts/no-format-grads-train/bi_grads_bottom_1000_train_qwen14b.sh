#!/bin/bash
#SBATCH --job-name=random_cleaned_train_qwen_2_5_14B
#SBATCH --partition=AI4Good_P
#SBATCH --gres=gpu:1

python ../../train_qwen14b.py --data_path /mnt/petrelfs/luzhenghao/safe_useful/grads_sort_results/alpaca_no_format_grads_bi_bottom_1000.json --output_path /mnt/petrelfs/luzhenghao/safe_useful/qwen_2_5_14B_output/alpaca_no_format_grads_bi_bottom_1000