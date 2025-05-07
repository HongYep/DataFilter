import torch
import gc
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import random
import numpy as np
use_logits = False
model_id = '/mnt/petrelfs/share_data/safety_verifier/models/Qwen2.5-7B-Instruct'
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             torch_dtype=torch.bfloat16,
                                             output_attentions=True,
                                             output_hidden_states=True,
                                             device_map="auto")
peft_model_id = 'qwen_2_5_7B_output/alpaca-gpt4_sample_1000/checkpoint-186'
peft_model = AutoModelForCausalLM.from_pretrained(peft_model_id,
                                             torch_dtype=torch.bfloat16,
                                             output_attentions=True,
                                             output_hidden_states=True,
                                             device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)
safe_data = []
# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("0.0.0.0", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

def data_process(data, safe_data = False, is_peft = False):
    if safe_data:
        if data['input'] == '':
            message = [
                {"role": "user", "content": f"{data['instruction']}"},
                # {"role": "assistant", "content": data['output']},
            ]
        else:
            message = [
                {"role": "user", "content": f"{data['instruction']}\n{data['input']}"},
                # {"role": "assistant", "content": data['output']},
            ]
    else:
        try:
            message = data['messages'][:1]
        except:
            message = [
                {"role": "user", "content": data},
            ]
    with torch.inference_mode():
        input_ids = tokenizer.apply_chat_template(
            message,
            return_tensors="pt"
        ).to(model.device)
        if is_peft:
            input_ids.to(peft_model.device)
            res = peft_model(input_ids)
        else:
            res = model(input_ids)
        # print(res)    
        hidden_states = torch.stack(res.hidden_states[1:]).detach().cpu()
    del input_ids, res
    gc.collect()
    torch.cuda.empty_cache()
    return hidden_states

with open('/mnt/petrelfs/share_data/lihao1/data/train_data/alpaca-gpt4.json','r') as f:
    train_safe_data =json.load(f)

with open('data/uncensor_circut.json','r') as f:
    unsafe_circut = json.load(f)

data_path = ['safe_test/harmbench.csv', 'safe_test/phi/total.csv', 'safe_test/directHarm4.csv']    

test_unsafe_benches = [pd.read_csv('safe_test/harmbench.csv')['Goal'].to_list(),
                       pd.read_csv('safe_test/phi/total.csv')['Goal'].to_list(),
                       pd.read_csv('safe_test/directHarm4.csv')['Goal'].to_list()]

with open('qwen_bestlayer.txt', 'a') as f:
    print(model_id, file=f)                       
for i in range(len(test_unsafe_benches)):
    test_unsafe_bench = test_unsafe_benches[i]
    with open('qwen_bestlayer.txt', 'a') as f:
        print(data_path[i], file=f)   
    for _ in range(5):
        safe_data = random.sample(train_safe_data, 100)
        test_safe_data = random.sample(train_safe_data, 100)
        unsafe_data = random.sample(unsafe_circut, 100)
        # test_unsafe_data = pd.read_csv('safe_test/directHarm4.csv')['Goal'].to_list()
        test_unsafe_data = random.sample(test_unsafe_bench, 100)
        safe_datas = [data_process(safe_data_item, True) for safe_data_item in tqdm(safe_data ,desc='safe_datas')]
        test_safe_datas = [data_process(safe_data_item, True, is_peft= True) for safe_data_item in tqdm(test_safe_data, desc='test_safe_datas')]
        unsafe_datas = [data_process(unsafe_data_item) for unsafe_data_item in tqdm(unsafe_data, desc='unsafe_datas')]
        test_unsafe_datas = [data_process(unsafe_data_item,  is_peft= True) for unsafe_data_item in tqdm(test_unsafe_data, desc='test_unsafe_datas')]

        acc_score = []
        # 设置噪声的强度（标准差）
        # noise_std = 0.5  # 可以根据需要调整这个值

        for index in range(0, 28):
            X_train = []
            for sa in safe_datas:
                X_train.append(sa[index].squeeze()[-1].to(torch.float).numpy())
            for unsa in unsafe_datas:
                X_train.append(unsa[index].squeeze()[-1].to(torch.float).numpy())
            
            # 将X_train转换为numpy数组
            X_train = np.array(X_train)
            
            # 添加高斯噪声
            # noise = np.random.normal(0, noise_std, size=X_train.shape)
            # X_train_noisy = X_train + noise
            
            Y_train = np.array([1] * len(safe_datas) + [0] * len(unsafe_datas))
            prob_model = LogisticRegression()
            # model.fit(X_train_noisy, Y_train)  # 使用带噪声的数据训练
            prob_model.fit(X_train, Y_train)
            
            X_test = []
            for unsa in test_unsafe_datas:
                X_test.append(unsa[index].squeeze()[-1].to(torch.float).numpy())
            for sa in test_safe_datas:
                X_test.append(sa[index].squeeze()[-1].to(torch.float).numpy())
            Y_test = np.array([0] * len(test_unsafe_datas)+[1] * len(test_safe_datas))
            y_pred = prob_model.predict_proba(X_test)[:, 1]
            y_pred_labels = (y_pred > 0.5).astype(int)
            accuracy_ = round(accuracy_score(Y_test, y_pred_labels), 4)
            acc_score.append(accuracy_)
            # print(f"layer {index} accuracy: {accuracy_}")
        with open('qwen_bestlayer.txt', 'a') as f:
            print(acc_score, file=f) 
# with open('data/train_data/alpaca-gpt4.json','r') as f:
#     train_data = json.load(f)
# train_datas = [data_process(train_data_item, True) for train_data_item in tqdm(train_data ,desc='train_datas')]
# data_X = []
# for data in train_datas:
#     data_X.append(data[index].squeeze()[-1].numpy())
# train_preds = model.predict_proba(train_data)[:,1]
# final_data = []
# for data, train_pred in zip(train_data, train_preds):
#     final_data.append({
#         **data,
#         'score':train_pred
#     })
# with open('data/train_data/alpaca-gpt4_prob_score.json','w') as f:
#     json.dump(final_data, f, indent=4)