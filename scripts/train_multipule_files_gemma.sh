#!/bin/bash

files=(
    'alpaca-gpt4_wild_score'
    'dolly_wild_score'
    'gsm8k_wild_score'
    'HealthCareMagic_wild_score'
    'magicoder_wild_score'
)

data_dir="/mnt/petrelfs/luzhenghao/safe_useful/wild_score_results"
# data_dir="/mnt/petrelfs/luzhenghao/safe_useful/data"
output_dir="/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_12b_output"

for file in "${files[@]}"; do
    echo "开始处理 ${file}_bottom_1000"
    cat > "debug_${file}.sh" << EOF
#!/bin/bash
#SBATCH --job-name=${file}_bottom_1000
#SBATCH --partition=AI4Good_P
#SBATCH --gres=gpu:1
#SBATCH --output=${output_dir}/${file}_bottom_1000.log
python /mnt/petrelfs/luzhenghao/safe_useful/train_gemma_12b.py --data_path ${data_dir}/${file}_bottom_1000.json --output_path ${output_dir}/${file}_bottom_1000
EOF
    sbatch "debug_${file}.sh"

    echo "开始处理 ${file}_top_1000"
    cat > "debug_${file}.sh" << EOF
#!/bin/bash
#SBATCH --job-name=${file}_top_1000
#SBATCH --partition=AI4Good_P
#SBATCH --gres=gpu:1
#SBATCH --output=${output_dir}/${file}_top_1000.log
python /mnt/petrelfs/luzhenghao/safe_useful/train_gemma_12b.py --data_path ${data_dir}/${file}_top_1000.json --output_path ${output_dir}/${file}_top_1000
EOF
    sbatch "debug_${file}.sh"
done