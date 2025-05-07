#!/bin/bash
#SBATCH --job-name=llama_bi_res_25226_HealthCareMagic_avg100_mean_top_1000
#SBATCH --partition=AI4Good_L
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/petrelfs/luzhenghao/safe_useful/llama_output/llama_bi_res_25226_HealthCareMagic_avg100_mean_top_1000.log
python /mnt/petrelfs/luzhenghao/safe_useful/train_llama.py --data_path /mnt/petrelfs/luzhenghao/safe_useful/llama_sort_results/llama_bi_res_25226_HealthCareMagic_avg100_mean_top_1000.json --output_path /mnt/petrelfs/luzhenghao/safe_useful/llama_output/llama_bi_res_25226_HealthCareMagic_avg100_mean_top_1000
