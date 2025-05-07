#!/bin/bash

files=(
    'alpaca-gpt4_sample_1000'
    'dolly_sample_1000'
    # 'gsm8k_sample_1000'
    # 'HealthCareMagic_sample_1000'
    # 'magicoder_sample_1000'
)

data_dir="/mnt/petrelfs/luzhenghao/safe_useful/sample_1000_results"
output_dir="/mnt/petrelfs/luzhenghao/safe_useful/llama3_output"

for file in "${files[@]}"; do
    echo "开始处理 ${file}"
    cat > "debug_${file}.sh" << EOF
#!/bin/bash
#SBATCH --job-name=${file}
#SBATCH --partition=AI4Good_P
#SBATCH --gres=gpu:1
#SBATCH --output=${output_dir}/${file}.log
python /mnt/petrelfs/luzhenghao/safe_useful/train_llama3.py --data_path ${data_dir}/${file}.json --output_path ${output_dir}/${file}
EOF
    sbatch "debug_${file}.sh"
done    