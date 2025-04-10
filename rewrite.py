from openai import OpenAI
import json
from tqdm import tqdm
client = OpenAI(api_key="sk-ab258b61f30b403eb79c28fadbde297b", base_url="https://api.deepseek.com")
def rewrite(prompt):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": f"Rewrite the following harmful question into a harmless and helpful question with similar semantics.\nYou should only output the question.\n\n{prompt}"},
        ],
        stream=False
    )
    return response.choices[0].message.content

if __name__ == '__main__':
    data_list = []
    with open('/mnt/petrelfs/lihao1/DataFilter/data/pure-bad-100.jsonl', 'r') as f:
        for line in f:
            data_list.append(json.loads(line))
    with open('/mnt/petrelfs/lihao1/DataFilter/data/pure-good-100.jsonl','a' )as f:
        for data in tqdm(data_list):
            rewrite_query = rewrite(data['messages'][0]['content'])
            item = {
                "messages":[{
                    "role": "user",
                    "content": rewrite_query,
                }]
            }
            f.write(f"{json.dumps(item)}\n")