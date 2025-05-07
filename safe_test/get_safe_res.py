from transformers import AutoModelForCausalLM, AutoTokenizer, Gemma3ForCausalLM
import json
import pandas as pd
import torch
from tqdm import tqdm
import argparse
        
        
def safe_test_peft_models(bench='direct'):
    model_path = '/mnt/petrelfs/share_data/ai4good_shared/models/google/gemma-2-9b-it'

    # '/mnt/petrelfs/share_data/safety_verifier/models/gemma-3-12b-it'

    # '/mnt/petrelfs/share_data/safety_verifier/models/Qwen2.5-7B-Instruct'
    
    # '/mnt/petrelfs/share_data/safety_verifier/models/Qwen2.5-14B-Instruct'
    
    # '/mnt/petrelfs/share_data/safety_verifier/models/Llama-3.1-8B-Instruct'

    file = '/mnt/petrelfs/luzhenghao/safe_useful/data/category_samples.json'

    with open(file, 'r') as f:
        data = json.load(f)

    if 'gemma' in model_path.lower():
        model = Gemma3ForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    final_list = []
    for item in tqdm(data):
        messages = [
            {
                'role':'user',
                'content':item['prompt'],
            }
        ]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        
        outputs = model.generate(
            input_ids,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0,
        )
        # print(outputs[0][input_ids.shape[-1]:])
        response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        # print(response)
        item['gemma_2_9b_output'] = response
        final_list.append(item)
        with open(f'/mnt/petrelfs/luzhenghao/safe_useful/data/category_samples.json','w') as f:
            json.dump(final_list, f, indent=4)

if __name__ == '__main__':
    safe_test_peft_models()
