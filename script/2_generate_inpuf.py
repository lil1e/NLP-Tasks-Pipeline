
import json
import os
import yaml
#-------------------------------配置信息------------------------------------

#配置config.yaml文件路径
config_file_path = '/home/zjy/14_project/config.yaml'

#---------------------------------------------------------------------------

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config(config_file_path)['2_generate_inpuf']

extractd_set = config['extractd_set']

# 模型的输入是jsonl文件！
model_inpuf = config['model_inpuf']

shot_num = config['shot_num']
task_type = config['task_type']
language = config['language']

# #-----------------------------------数据示例----------------------------
'''
输入：
extractd_set样本示例:

{
    "conversation_id": 1,
    "human": "Output the chemical-induced disease relations in the following text:\nNaloxone reverses the antihypertensive effect of clonidine. In unanesthetized, spontaneously hypertensive rats the decrease in blood pressure and heart rate produced by intravenous clonidine, 5 to 20 micrograms/kg, was inhibited or reversed by nalozone, 0.2 to 2 mg/kg. The hypotensive effect of 100 mg/kg alpha-methyldopa was also partially reversed by naloxone. Naloxone alone did not affect either blood pressure or heart rate. In brain membranes from spontaneously hypertensive rats clonidine, 10(-8) to 10(-5) M, did not influence stereoselective binding of [3H]-naloxone (8 nM), and naloxone, 10(-8) to 10(-4) M, did not influence clonidine-suppressible binding of [3H]-dihydroergocryptine (1 nM). These findings indicate that in spontaneously hypertensive rats the effects of central alpha-adrenoceptor stimulation involve activation of opiate receptors. As naloxone and clonidine do not appear to interact with the same receptor site, the observed functional antagonism suggests the release of an endogenous opiate by clonidine or alpha-methyldopa and the possible role of the opiate in the central control of sympathetic tone.",
    "example": [
        {
            "conversation_id": 348,
            "human": "According to the given text, extract the chemical-induced disease relations from the text: \"Sodium status influences chronic amphotericin B nephrotoxicity in rats. The nephrotoxic potential of amphotericin B (5 mg/kg per day intraperitoneally for 3 weeks) has been investigated in salt-depleted, normal-salt, and salt-loaded rats. In salt-depleted rats, amphotericin B decreased creatinine clearance linearly with time, with an 85% reduction by week 3. In contrast, in normal-salt rats creatinine clearance was decreased but to a lesser extent at week 2 and 3, and in salt-loaded rats creatinine clearance did not change for 2 weeks and was decreased by 43% at week 3. All rats in the sodium-depleted group had histopathological evidence of patchy tubular cytoplasmic degeneration in tubules that was not observed in any normal-salt or salt-loaded rat. Concentrations of amphotericin B in plasma were not significantly different among the three groups at any time during the study. However, at the end of 3 weeks, amphotericin B levels in the kidneys and liver were significantly higher in salt-depleted and normal-salt rats than those in salt-loaded rats, with plasma/kidney ratios of 21, 14, and 8 in salt-depleted, normal-salt, and salt-loaded rats, respectively. In conclusion, reductions in creatinine clearance and renal amphotericin B accumulation after chronic amphotericin B administration were enhanced by salt depletion and attenuated by sodium loading in rats.\"",
            "assistant": "chemical-induced disease relations: [amphotericin B, nephrotoxicity]; [amphotericin B, nephrotoxic]"
        }
    ],
    "assistant": "chemical-induced disease relations: [alpha-methyldopa, hypotensive]"
}
'''
#------------------------------prompt template---------------------------

# #NER-zh
#     '''
#     任务类型:命名实体识别

#     例子:
#     input:
#     {shot_text}
#     output
#     {shot_label}

#     新任务:
#     {input_text}

#     '''
# #NER-en
# ...

def generate_kshot_prompt(query_text, examples, task_type, k=shot_num, language='en'):

    """
    生成k-shot任务的prompt。

    参数:
    examples: list of dicts - 包含k个示例的列表，每个示例包含 'human' 和 'assistant' 字段
    task_type: str - 任务类型，可以是 'ner', 're', 'tc', 'qa'
    k: int - k-shot示例的数量
    language: str - 语言偏好，'en' 为英文，'zh' 为中文

    返回:
    prompt: str - 格式化后的prompt
    """

    # 定义语言特定的任务类型描述
    task_descriptions = {
        'ner': "命名实体识别" if language == 'zh' else "Named Entity Recognition",
        're': "关系抽取" if language == 'zh' else "Relation Extraction",
        'tc': "文本分类" if language == 'zh' else "Text Classification",
        'qa': "问答" if language == 'zh' else "Question Answering"
    }

    # 检查任务类型是否有效
    if task_type not in task_descriptions:
        raise ValueError(f"不支持的任务类型: {task_type}")

    # 获取任务类型描述
    task_description = task_descriptions[task_type]

   
    # 初始化prompt
    prompt = f"任务类型:{task_description}\n\n例子:\n" if language == 'zh' else f"Task Type:{task_description}\n\nExample:\n"

    # 添加k个示例
    for i, example in enumerate(examples[:k], start=1):
        human_text = example['human']
        assistant_text = example['assistant']
        prompt += f"{i}. input: {human_text}\noutput: {assistant_text}\n\n"

    # 添加新任务的提示，根据语言偏好设置
    new_task_prompt = f"新任务:\n{query_text}" if language == 'zh' else f"New Task:\n{query_text}"
    prompt += new_task_prompt

    return prompt


#-----------------------------------生成model_inpuf----------------------------

with open(extractd_set, encoding='utf-8') as f:
    dataset = json.load(f)
json_data = []
for data in dataset:
    #构建示例
    query_text = data['human']
    examples = data['example']

    assistant_text = data['assistant']
    human_text = generate_kshot_prompt(query_text, examples, task_type, shot_num, language)
    
    conversation =[
        {
        "human": human_text,
        "assistant": assistant_text
        }
    ]

    new_item = {
        "conversation": conversation
    }
    json_data.append(new_item)
    
with open(model_inpuf, 'w', encoding='utf-8') as f:
    for item in json_data:
        # 将每个JSON对象转换为JSON字符串并写入单独的一行
        f.write(json.dumps(item,ensure_ascii=False) + '\n')

#测试
# print(json.dumps(json_data[0], ensure_ascii=False, indent=4))
print(f"model_inpuf saved to {model_inpuf}")