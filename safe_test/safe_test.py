from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, LlavaForConditionalGeneration, Gemma3ForCausalLM
import json
import pandas as pd
import torch
from tqdm import tqdm
import argparse
# model = AutoModelForCausalLM.from_pretrained('output/Llama-3-code/checkpoint-312', torch_dtype=torch.bfloat16, device_map="auto")
# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("0.0.0.0", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass
def safe_test_llama():
    # model_id = ''
    model_id = '/mnt/petrelfs/lihao1/trustai/models/Meta-Llama-3-8B-Instruct'
    # peft_id = 'output/Llama-3-code/checkpoint-312'
    peft_id = '/mnt/petrelfs/lihao1/vlmmoe/output/Llama-3-med_safe_100/checkpoint-462'
    # peft_id = '/mnt/petrelfs/lihao1/LLaMA-Factory/saves/llama3-8b/lora/sft/checkpoint-841'
    # peft_id= model_id
    model = AutoModelForCausalLM.from_pretrained(peft_id, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # goals = [item['instruction'] for item in data]
    # goals = pd.read_csv('safe_test/advbench.csv')['Goal'].to_list()
    # goals = pd.read_csv('safe_test/directHarm4.csv')['Goal'].to_list()
    # goals = pd.read_csv('safe_test/harmbench.csv')['Goal'].to_list()
    goals = pd.read_csv('safe_test/phi/total.csv')['Goal'].to_list()
    final_list = []
    for goal in tqdm(goals):
        messages = [
            {
                'role':'user',
                'content':goal,
            }
        ]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
        
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        outputs = model.generate(
            input_ids,
            max_new_tokens=64,
            eos_token_id=terminators,
            do_sample=False,
        )
        
        response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        final_list.append({
            'instruction':goal,
            'output':response
        })
        
        with open('safe_test/results/gsm8k-lora-llama-phi-output.json', 'w') as f:
            json.dump(final_list, f, indent=4)

def safe_test_llava():
    from PIL import Image
    model_id = '/mnt/petrelfs/lihao1/trustai/huxuhao/models/llava-1.5-7b-hf'
    peft_list = [
        # 'DocVQA',
        # 'TabMWP',
        # 'PMC-VQA',
        # 'ScienceQA',
        "GeoQA+"
    ]
    for peft in peft_list:
        peft_id = f'/mnt/hwfile/trustai/wangxinyuan/lora/llava1_5-7b-hf/{peft}/checkpoint-1686'
        model = LlavaForConditionalGeneration.from_pretrained(peft_id, torch_dtype=torch.bfloat16, device_map="auto")
        processor = AutoProcessor.from_pretrained(model_id)
        with open('safe_test/MM-processed_questions/total.json','r') as f:
            mm_safe_questions = json.load(f)
        final_list = []
        for question in tqdm(mm_safe_questions):
            image_path = '/mnt/petrelfs/lihao1/vlmmoe/safe_test/MM-img/'+question['sd-typo-image']
            conversation = [
                {
                    'role':'user',
                    'content':[
                        {
                            'type':'text',
                            'text':question['Rephrased Question']
                        },
                        {
                            'type':'image',
                        }
                    ]
                }
            ]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            raw_image = Image.open(image_path)
            inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(model.device)
            output = model.generate(**inputs, max_new_tokens = 96, do_sample=False)
            response = processor.decode(output[0], skip_special_tokens=True).split('ASSISTANT: ')[1]
            final_list.append({
                'instruction':question['Rephrased Question'],
                'image':image_path,
                'output':response,
            })
            with open(f'safe_test/results/llava-{peft}-mm-output.json', 'w') as f:
                json.dump(final_list, f, indent=4)

def safe_test_qwen_vl():
    from qwen_vl_utils import process_vision_info
    model_id = "/mnt/petrelfs/lihao1/trustai/lijun/models/Qwen2-VL-7B-Instruct"
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained(model_id)
    with open('safe_test/MM-processed_questions/total.json','r') as f:
        mm_safe_questions = json.load(f)
    final_list = []
    for question in tqdm(mm_safe_questions):
        image_path = '/mnt/petrelfs/lihao1/vlmmoe/safe_test/MM-img/'+question['sd-typo-image']
        messages = [
            {
                'role':'user',
                'content':[
                    {
                        'type':'text',
                        'text':question['Rephrased Question']
                    },
                    {
                        'type':'image',
                        "image":image_path
                    }
                ]
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        final_list.append({
            'instruction':question['Rephrased Question'],
            'image':image_path,
            'output':output_text[0],
        })
        with open('safe_test/results/qwen-mm-output.json', 'w') as f:
            json.dump(final_list, f, indent=4)

def safe_test_peft_models(bench='direct'):
    peft_models = {
        # 'gemma_3_4b_bi_res_logits_avg100_mean_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_4b_output/gemma_3_4b_bi_res_logits_avg100_mean_bottom_1000/checkpoint-186',
        # 'gemma_3_4b_bi_res_logits_avg100_mean_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_4b_output/gemma_3_4b_bi_res_logits_avg100_mean_top_1000/checkpoint-186',
        # 'gemma_3_4b_bi_res_rep_avg100_mean_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_4b_output/gemma_3_4b_bi_res_rep_avg100_mean_bottom_1000/checkpoint-186',
        # 'gemma_3_4b_bi_res_rep_avg100_mean_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_4b_output/gemma_3_4b_bi_res_rep_avg100_mean_top_1000/checkpoint-186',
        'gemma_3_12b_bi_res_logits_avg100_mean_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_12b_output/gemma_3_12b_bi_res_logits_avg100_mean_bottom_1000/checkpoint-186',
        'gemma_3_12b_bi_res_logits_avg100_mean_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_12b_output/gemma_3_12b_bi_res_logits_avg100_mean_top_1000/checkpoint-186',
        # 'llama_bi_res_logits_avg100_mean_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/llama_output/llama_bi_res_logits_avg100_mean_bottom_1000/checkpoint-186',
        # 'llama_bi_res_logits_avg100_mean_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/llama_output/llama_bi_res_logits_avg100_mean_top_1000/checkpoint-186',
        # 'llama_bi_res_rep_avg100_mean_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/llama_output/llama_bi_res_rep_avg100_mean_bottom_1000/checkpoint-186',
        # 'llama_bi_res_rep_avg100_mean_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/llama_output/llama_bi_res_rep_avg100_mean_top_1000/checkpoint-186',
        # 'qwen_2_5_7B_bi_res_logits_avg100_mean_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/qwen_2_5_7B_output/qwen_2_5_7B_bi_res_logits_avg100_mean_bottom_1000/checkpoint-186',
        # 'qwen_2_5_7B_bi_res_logits_avg100_mean_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/qwen_2_5_7B_output/qwen_2_5_7B_bi_res_logits_avg100_mean_top_1000/checkpoint-186',
        # 'qwen_2_5_14B_bi_res_logits_avg100_mean_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/qwen_2_5_14B_output/qwen_2_5_14B_bi_res_logits_avg100_mean_bottom_1000/checkpoint-186',
        # 'qwen_2_5_14B_bi_res_logits_avg100_mean_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/qwen_2_5_14B_output/qwen_2_5_14B_bi_res_logits_avg100_mean_top_1000/checkpoint-186',

    }
    
    def get_tokenizer_path(peft_path):
        if 'gemma_3_4b' in peft_path.lower():
            return '/mnt/petrelfs/share_data/safety_verifier/models/gemma-3-4b-it'
        
        elif 'gemma_3_12b ' in peft_path.lower():
            return '/mnt/petrelfs/share_data/safety_verifier/models/gemma-3-12b-it'

        elif 'qwen_2_5_7b' in peft_path.lower():
            return '/mnt/petrelfs/share_data/safety_verifier/models/Qwen2.5-7B-Instruct'
        
        elif 'qwen_2_5_14b' in peft_path.lower():
            return '/mnt/petrelfs/share_data/safety_verifier/models/Qwen2.5-14B-Instruct'
        
        return '/mnt/petrelfs/share_data/safety_verifier/models/Llama-3.1-8B-Instruct'
    
    def get_goals(bench):
        goals = {
            'test':pd.read_csv('/mnt/petrelfs/luzhenghao/safe_useful/safe_test/test.csv')['Goal'].to_list(),
            'direct':pd.read_csv('/mnt/petrelfs/luzhenghao/safe_useful/safe_test/directHarm4.csv')['Goal'].to_list(),
            'harm':pd.read_csv('/mnt/petrelfs/luzhenghao/safe_useful/safe_test/harmbench.csv')['Goal'].to_list(),
            'phi':pd.read_csv('/mnt/petrelfs/luzhenghao/safe_useful/safe_test/phi/total.csv')['Goal'].to_list()
        }
        return goals[bench]
    
    goals = get_goals(bench)
    for peft_id, peft_path in tqdm(peft_models.items()):
        if 'gemma' in peft_path.lower():
            model = Gemma3ForCausalLM.from_pretrained(peft_path, torch_dtype=torch.bfloat16, device_map="auto")
        else:
            model = AutoModelForCausalLM.from_pretrained(peft_path, torch_dtype=torch.bfloat16, device_map="auto")
        print(get_tokenizer_path(peft_path))
        tokenizer = AutoTokenizer.from_pretrained(get_tokenizer_path(peft_path))
        final_list = []
        for goal in tqdm(goals):
            messages = [
                {
                    'role':'user',
                    'content':goal,
                }
            ]
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(model.device)
            
            outputs = model.generate(
                input_ids,
                max_new_tokens=160,
                do_sample=False,
                temperature=0.0,
            )
            # print(outputs[0][input_ids.shape[-1]:])
            response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
            final_list.append({
                'instruction':goal,
                'output':response
            })
            with open(f'./med_results/{peft_id}-{bench}.json','w') as f:
                json.dump(final_list, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bench', type=str, default='direct')
    args = parser.parse_args()
    # safe_test_qwen_vl()
    # safe_test_llama()
    safe_test_peft_models(args.bench)
    # safe_test_llava()
    # safe_test_llama_code()