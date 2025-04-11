#!/bin/bash
#SBATCH --job-name=alpaca_data_cleaned_moderation_score_bottom_1000
#SBATCH --partition=AI4Good_S1
#SBATCH --gres=gpu:1
#SBATCH --output=../llama_output/alpaca_data_cleaned_moderation_score_bottom_1000.log
python ../train_llama_s.py --data_path ../data/alpaca_data_cleaned_moderation_score_bottom_1000.json --output_path ../llama_output/alpaca_data_cleaned_moderation_score_bottom_1000