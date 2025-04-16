#!/bin/bash
#SBATCH --job-name=magicoder_sample_1000
#SBATCH --partition=AI4Good_S1
#SBATCH --gres=gpu:1
#SBATCH --output=../llama_output/magicoder_sample_1000.log 
python ../train_llama_s.py --data_path ../llama_sort_results/magicoder_sample_1000.json --output_path ../llama_output/magicoder_sample_1000