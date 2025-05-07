#!/bin/bash
#SBATCH --job-name=pubmed_qa
#SBATCH --partition=AI4Good_L1_p
#SBATCH --gres=gpu:1
lm_eval --model hf     --model_args pretrained=/mnt/petrelfs/share_data/ai4good_shared/models/google/gemma-2-9b-it,peft=/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/gemma2/gemma2_pubmed_remove_top_2000     --tasks pubmedqa     --device cuda     --batch_size 16     --num_fewshot 0     --trust_remote_code
