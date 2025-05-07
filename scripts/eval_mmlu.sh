lm_eval --model hf \
    --model_args pretrained=/mnt/petrelfs/lihao1/trustai/share/models/meta-llama/Llama-3.1-8B-Instruct \
    --tasks mmlu \
    --device cuda:0 \
    --batch_size 4