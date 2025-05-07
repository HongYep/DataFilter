#!/bin/bash

model="/mnt/petrelfs/share_data/safety_verifier/models/Meta-Llama-3-8B-Instruct"
base_dir="/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/llama3"
pefts=(
    'llama3_gsm8k_bottom_1000/checkpoint-96'
    'llama3_magicoder_remove_top_2000'
    'llama3_magicoder_remove_top_2000_d'
    'llama_score_magicoder_remove_top_2000'
    'wild_magicoder_remove_top_2000'
)



for peft in "${pefts[@]}"; do
    echo "开始处理 ${peft}"
    cat > "debug_${peft}.sh" << EOF
#!/bin/bash
#SBATCH --job-name=eval_gsm8k
#SBATCH --partition=AI4Good_L
#SBATCH --gres=gpu:1
lm_eval --model hf \
    --model_args pretrained=${model},peft=${base_dir}/${peft} \
    --tasks gsm8k \
    --device cuda \
    --batch_size 16 \
    --apply_chat_template \
    --num_fewshot 0
EOF
    sbatch "debug_${peft}.sh"
done

# pretrained=/mnt/petrelfs/share_data/safety_verifier/models/Llama-3.1-8B-Instruct,peft=/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/llama3.1/llama_gsm8k_bottom_1000
# pretrained=/mnt/petrelfs/share_data/safety_verifier/models/Qwen2.5-7B-Instruct,peft=/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/qwen2.5-instruct/qwen_gsm8k_bottom_1000
# pretrained=/mnt/petrelfs/share_data/safety_verifier/models/Meta-Llama-3-8B-Instruct,peft=/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/llama3/llama3_gsm8k_bottom_1000
# pretrained=/mnt/petrelfs/share_data/ai4good_shared/models/google/gemma-2-9b-it,peft=/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/gemma2/gemma2_gsm8k_bottom_1000