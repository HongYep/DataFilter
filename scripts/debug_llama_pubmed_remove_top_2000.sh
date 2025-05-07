#!/bin/bash
#SBATCH --job-name=pubmed_qa
#SBATCH --partition=AI4Good_L1_p
#SBATCH --gres=gpu:1
lm_eval --model hf     --model_args pretrained=/mnt/petrelfs/share_data/safety_verifier/models/Llama-3.1-8B-Instruct,peft=/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/llama3.1/llama_pubmed_remove_top_2000     --tasks pubmedqa     --device cuda     --batch_size 16     --num_fewshot 0     --trust_remote_code
