#!/bin/bash
#SBATCH --job-name=eval_humaneval
#SBATCH --partition=AI4Good_L
#SBATCH --gres=gpu:1
lm_eval --model hf     --model_args pretrained=/mnt/petrelfs/share_data/safety_verifier/models/Meta-Llama-3-8B-Instruct,peft=/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/llama3/llama3_magicoder_bottom_1000/checkpoint-96     --tasks humaneval_instruct     --device cuda     --batch_size 16     --num_fewshot 0     --apply_chat_template     --confirm_run_unsafe_code
