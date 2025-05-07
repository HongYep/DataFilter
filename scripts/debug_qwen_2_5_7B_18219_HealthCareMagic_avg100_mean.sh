#!/bin/bash
#SBATCH --job-name=qwen_2_5_7B_18219_HealthCareMagic_avg100_mean_top_1000
#SBATCH --partition=AI4Good_L
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/petrelfs/luzhenghao/safe_useful/qwen_2_5_7B_output/qwen_2_5_7B_18219_HealthCareMagic_avg100_mean_top_1000.log
python /mnt/petrelfs/luzhenghao/safe_useful/train_qwen.py --data_path /mnt/petrelfs/luzhenghao/safe_useful/qwen_2_5_7B_sort_results/qwen_2_5_7B_18219_HealthCareMagic_avg100_mean_top_1000.json --output_path /mnt/petrelfs/luzhenghao/safe_useful/qwen_2_5_7B_output/qwen_2_5_7B_18219_HealthCareMagic_avg100_mean_top_1000
