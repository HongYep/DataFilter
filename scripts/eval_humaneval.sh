lm_eval --model hf \
    --model_args pretrained=/mnt/petrelfs/lihao1/trustai/share/models/meta-llama/Llama-3.1-8B-Instruct,peft=/mnt/petrelfs/lihao1/DataFilter/llama_output/HealthCareMagic_sample_1000/checkpoint-100 \
    --tasks humaneval \
    --device cuda:0 \
    --batch_size auto \
    --confirm_run_unsafe_code