import torch.nn as nn
from datasets import Dataset
from peft import  LoraConfig, TaskType, get_peft_model
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    Gemma3ForCausalLM
)
import json
import os
import argparse

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

model = Gemma3ForCausalLM.from_pretrained('/mnt/petrelfs/share_data/safety_verifier/models/gemma-3-4b-it', device_map="auto", trust_remote_code = True)
model.enable_input_require_grads()  # 开启梯度检查点
tokenizer = AutoTokenizer.from_pretrained('/mnt/petrelfs/share_data/safety_verifier/models/gemma-3-4b-it', use_fast=False, trust_remote_code = True)

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("0.0.0.0", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

def alpaca_process_func(example):
    MAX_LENGTH = 2048
    input_ids, attention_mask, labels = [], [], []
    if example['input'] == '':
        message = [
            {
                "role": "user",
                "content":[
                    {
                        "type": "text",
                        "text": f"{example['instruction']}"
                    }
                ]
             },
        ]
    else:
        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text":f"{example['instruction']}\n{example['input']}"
                    }
                ]
            },
        ]
    insturction = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=True, return_dict=True)
    response = tokenizer(f"{example['output']}<end_of_turn>\n", add_special_tokens=False)
    input_ids = insturction['input_ids'] + response['input_ids'] + [tokenizer.pad_token_id]
    attention_mask = insturction['attention_mask'] + response['attention_mask'] + [1]
    labels = [-100] * len(insturction['input_ids']) + response['input_ids'] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='gemma_3_4b_sort_results/gemma_3_4b_bi_res_logits_avg100_mean_top_1000.json')
    parser.add_argument('--output_path', type=str, default='gemma_3_4b_output/gemma_3_4b_bi_res_logits_avg100_mean_top_1000')
    args = parser.parse_args()
    train_json_path = args.data_path
    train_ds = Dataset.from_json(train_json_path)
    train_dataset = train_ds.map(alpaca_process_func)
    # 配置LoRA
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,  # 训练模式
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        r=8,  # Lora 秩
    )
    peft_model = get_peft_model(model, config)

    # 配置Trainer
    os.environ["WANDB_PROJECT"]="Gemma"
    args = TrainingArguments(
        output_dir=args.output_path,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        logging_steps=1,
        num_train_epochs=3,
        save_steps=50,
        save_total_limit=1,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="wandb",
        warmup_ratio=0.1
    )
    trainer = Trainer(
        model=peft_model,
        args=args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    )
    trainer.train()