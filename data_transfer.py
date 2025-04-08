import json

def process_json_lines(input_file, output_file, modify_func):
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:

        for line in fin:
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                print(f"⚠️ 无效JSON行: {line}")
                continue

            modified_data = modify_func(data)

            fout.write(json.dumps(modified_data) + '\n')

def add_timestamp(data):
    if 'create_time' in data:
        data['timestamp'] = int(data['create_time'])
    return data

# 使用示例
if __name__ == "__main__":
    process_json_lines('input.jsonl', 'output.jsonl', add_timestamp)