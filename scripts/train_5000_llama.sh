#!/bin/bash

files=(
    # 'llama_bi_res_25226_alpaca-gpt4_avg100'
    # 'llama_bi_res_25226_dolly_avg100'
    'llama_bi_res_25226_gsm8k_avg100'
    'llama_bi_res_25226_magicoder_avg100'
    'llama_bi_res_25226_HealthCareMagic_avg100'
)

data_dir="/mnt/petrelfs/luzhenghao/safe_useful/llama_sort_results"
# data_dir="/mnt/petrelfs/luzhenghao/safe_useful/data"
output_dir="/mnt/petrelfs/luzhenghao/safe_useful/llama_output"

for file in "${files[@]}"; do
    echo "开始处理 ${file}_random_5000"
    cat > "debug_${file}.sh" << EOF
#!/bin/bash
#SBATCH --job-name=${file}_random_5000
#SBATCH --partition=AI4Good_P
#SBATCH --gres=gpu:1
#SBATCH --output=${output_dir}/${file}_random_5000.log
python /mnt/petrelfs/luzhenghao/safe_useful/train_llama.py --data_path ${data_dir}/${file}_random_5000.json --output_path ${output_dir}/${file}_random_5000
EOF
    sbatch "debug_${file}.sh"

    echo "开始处理 ${file}_lowest_5000"
    cat > "debug_${file}.sh" << EOF
#!/bin/bash
#SBATCH --job-name=${file}_lowest_5000
#SBATCH --partition=AI4Good_P
#SBATCH --gres=gpu:1
#SBATCH --output=${output_dir}/${file}_lowest_5000.log
python /mnt/petrelfs/luzhenghao/safe_useful/train_llama.py --data_path ${data_dir}/${file}_lowest_5000.json --output_path ${output_dir}/${file}_lowest_5000
EOF
    sbatch "debug_${file}.sh"
done