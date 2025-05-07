#!/bin/bash
#SBATCH --job-name=magicoder_sample_1000
#SBATCH --partition=AI4Good_P
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_12b_output/magicoder_sample_1000.log
python /mnt/petrelfs/luzhenghao/safe_useful/train_gemma_12b.py --data_path /mnt/petrelfs/luzhenghao/safe_useful/sample_1000_results/magicoder_sample_1000.json --output_path /mnt/petrelfs/luzhenghao/safe_useful/gemma_3_12b_output/magicoder_sample_1000
