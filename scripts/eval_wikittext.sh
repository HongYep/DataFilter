lm_eval --model hf \
    --model_args pretrained=/mnt/petrelfs/lihao1/trustai/share/models/meta-llama/Llama-3.1-8B-Instruct,peft=/mnt/petrelfs/lihao1/DataFilter/llama_output/alpaca_data_cleaned/checkpoint-3235 \
    --tasks wikitext \
    --device cuda:7 \
    --batch_size 4