#!/bin/bash

model="/mnt/petrelfs/share_data/safety_verifier/models/Qwen2.5-7B-Instruct"
# model="/mnt/petrelfs/share_data/safety_verifier/models/Llama-3.1-8B-Instruct"
# model="/mnt/petrelfs/share_data/safety_verifier/models/Meta-Llama-3-8B-Instruct"
# model="/mnt/petrelfs/share_data/ai4good_shared/models/google/gemma-2-9b-it"


base_dir="/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/qwen2.5-instruct"


pefts=(
    'pubmed'
    'qwen_pubmed_remove_top_2000'
)

    echo "开始处理 llama3"
    cat > "debug_llama3.sh" << EOF
#!/bin/bash
#SBATCH --job-name=eval_humaneval
#SBATCH --partition=AI4Good_L
#SBATCH --gres=gpu:1
lm_eval --model hf \
    --model_args pretrained=${model}\
    --tasks humaneval_instruct \
    --device cuda \
    --batch_size 16 \
    --num_fewshot 0 \
    --confirm_run_unsafe_code
EOF
    sbatch "debug_llama3.sh"


for peft in "${pefts[@]}"; do
    echo "开始处理 ${peft}"
    cat > "debug_${peft}.sh" << EOF
#!/bin/bash
#SBATCH --job-name=eval_humaneval
#SBATCH --partition=AI4Good_L
#SBATCH --gres=gpu:1
lm_eval --model hf \
    --model_args pretrained=${model},peft=${base_dir}/${peft} \
    --tasks humaneval_instruct \
    --device cuda \
    --batch_size 16 \
    --num_fewshot 0 \
    --apply_chat_template \
    --confirm_run_unsafe_code
EOF
    sbatch "debug_${peft}.sh"
done
