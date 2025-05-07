#!/bin/bash
#SBATCH --job-name=dolly_sample_1000
#SBATCH --partition=AI4Good_P
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/petrelfs/luzhenghao/safe_useful/llama3_output/dolly_sample_1000.log
python /mnt/petrelfs/luzhenghao/safe_useful/train_llama3.py --data_path /mnt/petrelfs/luzhenghao/safe_useful/sample_1000_results/dolly_sample_1000.json --output_path /mnt/petrelfs/luzhenghao/safe_useful/llama3_output/dolly_sample_1000
