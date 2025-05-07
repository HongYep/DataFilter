#!/bin/bash
#SBATCH --job-name=random_cleaned_train_llama
#SBATCH --partition=AI4Good_P
#SBATCH --gres=gpu:1

python ../../train_llama.py --data_path /mnt/petrelfs/luzhenghao/safe_useful/grads_sort_results/alpaca_no_format_grads_unsafe_bottom_1000.json --output_path /mnt/petrelfs/luzhenghao/safe_useful/llama_output/alpaca_no_format_grads_bottom_1000