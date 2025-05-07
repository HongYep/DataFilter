#!/bin/bash
#SBATCH --job-name=eval_humaneval
#SBATCH --partition=AI4Good_L
#SBATCH --gres=gpu:1
lm_eval --model hf     --model_args pretrained=/mnt/petrelfs/share_data/safety_verifier/models/Qwen2.5-7B-Instruct,peft=/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/qwen2.5-instruct/qwen_magicoder_remove_top_2000_d     --tasks humaneval_instruct     --device cuda     --batch_size 16     --num_fewshot 0     --apply_chat_template     --confirm_run_unsafe_code
