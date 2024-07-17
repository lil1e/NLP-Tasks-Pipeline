# encoding: utf-8
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from dataclasses import dataclass
from tqdm import tqdm
import yaml
#-------------------------------配置信息------------------------------------

#配置config.yaml文件路径
config_file_path = '/home/zjy/14_project/config.yaml'

#---------------------------------------------------------------------------
def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config(config_file_path)['3_val']

model_name = config['model_name']
device = config['device']
max_new_tokens = config['max_new_tokens']
top_p = config['top_p']
temperature = config['temperature']
repetition_penalty = config['repetition_penalty']

model_inpuf = config['model_inpuf']
pred_resulf = config['pred_resulf']

#------------------------------数据加载------------------------------------

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map=device
)
model.eval()

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)
tokenizer.pad_token_id = tokenizer.eod_id
tokenizer.bos_token_id = tokenizer.eod_id
tokenizer.eos_token_id = tokenizer.eod_id


# -----------------------------------模型推理-----------------------------

def chat(user_input):
    """
    LLM对话函数
    :param user_input:
    :return:
    """
    input_ids = tokenizer(user_input, return_tensors="pt", add_special_tokens=False).input_ids
    bos_token_id = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long)
    eos_token_id = torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long)
    user_input_ids = torch.concat([bos_token_id, input_ids, eos_token_id], dim=1)

    model_input_ids = user_input_ids.to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=model_input_ids, max_new_tokens=max_new_tokens, do_sample=True, top_p=top_p,
            temperature=temperature, repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eos_token_id
        )

    model_input_ids_len = model_input_ids.size(1)
    response_ids = outputs[:, model_input_ids_len:]

    response = tokenizer.batch_decode(response_ids)
    result = response[0].strip().replace(tokenizer.eos_token, "")

    return result


# 加载数据 开始预测

dataset = []

with open(model_inpuf, 'r', encoding='utf-8') as file:
    for line in file:
        # 解析每一行的JSON对象并添加到列表中
        json_object = json.loads(line)
        dataset.append(json_object)

json_data = []

for input_data in tqdm(dataset):
    user_input = input_data["conversation"][0]["human"]
    label = input_data["conversation"][0]["assistant"]

    pred = chat(user_input)

    json_data.append({"pred": pred, "label": label})

# 保存预测实体结果
with open(pred_resulf, 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=4)
