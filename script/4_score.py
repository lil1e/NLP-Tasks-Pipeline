# import argparse
import json
import re
from datetime import datetime
import yaml
#-----------------------------------------------------------------------------

#配置config.yaml文件路径
config_file_path = '/home/zjy/14_project/config.yaml'

#---------------------------------------------------------------------------

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config(config_file_path)['4_score']


# 脚本配置输入路径
notes = config['notes']
task_type = config['task_type']
pred_resulf = config['pred_resulf']
log_file = config['log_file']

#-----------------------------------------------------------------------------
# 定义不同任务类型的提取函数
def extract_qa(text):
    parts = text.split(":")
    extracted_content = parts[1].strip()
    return extracted_content

def extract_tc(text):
    parts = text.split("：", 1)
    # 确保列表至少有两个元素，如果分割后只有一个元素，则将第二部分设置为空字符串
    # 这是因为有少量结果输出：上述文本中没有相关实体
    if len(parts) == 1:
        parts.append("")
        # print(parts[1])
    
    extracted_content = parts[1].strip()
    return extracted_content

def extract_re(text):
    matches = re.findall(r'\[([^]]+)\]', text)
    extracted_list = [item.split(', ') for item in matches]
    return extracted_list

def get_pair_ner(text):
    res = []
    pred_res = text.split("\n")
    for scan in pred_res:
        for idx, char in enumerate(scan):
            if char == ":":
                category = scan[:idx].strip()
                entities = scan[idx+1:].strip().split(";")
                for e in entities:
                    res.append(f"{category}-{e.strip()}")
                break
    return res

# 定义评估函数
def evaluate(task_type, predictions_file):
    with open(predictions_file, encoding='utf-8') as f:
        raw_data = json.load(f)

    TP = 0
    FP = 0
    FN = 0

    extract_func = None
    if task_type == 'qa':
        extract_func = extract_qa
    elif task_type == 'tc':
        extract_func = extract_tc
    elif task_type == 're':
        extract_func = extract_re
    elif task_type == 'ner':
        extract_func = get_pair_ner
    else:
        raise ValueError("Unknown task type")

    for data in raw_data:
        pred = data['pred']
        label = data['label']

        predicted = extract_func(pred)
        true = extract_func(label)

        # 计算TP, FP, FN
        # 根据任务类型不同，这里的逻辑可能需要调整
        # 以下是一个通用的示例逻辑，具体实现可能需要根据任务特点进行调整
        for p in predicted:
            if p in true:
                TP += 1
            else:
                FP += 1

        for t in true:
            if t not in predicted:
                FN += 1

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1

#---------------------------------------------------------------------------
# 主函数
def main(task_type, predictions_file, log_file):
    # 根据任务类型选择输出precision或f1
    score_to_output = None
    if task_type == 'qa':
        score_to_output = 'accurancy'
    else:
        score_to_output = 'f1'

    precision, recall, f1 = evaluate(task_type, predictions_file)

    # 获取当前日期和时间
    current_time = datetime.now()
    # 格式化日期和时间为字符串
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

    # 准备日志信息
    log_message = f"Notes: {notes}\n" \
                  f"\tTask Type: {task_type}\n" \
                  f"\tDate and Time: {formatted_time}\n" \
                  f"\tFile: {predictions_file}\n" \
                  f"\t{score_to_output}: {f1 if task_type != 'qa' else precision}\n\n"  #经测试，accurancy值与precision值相等

    # 记录到文件
    with open(log_file, 'a', encoding='utf-8') as log:
        log.write(log_message)
        #log.write(f"Task Type: {task_type}\n\tFile: {predictions_file}\n\t{score_to_output}: {locals()[score_to_output]}\n\n")

    # 打印到控制台
    print(f"Task Type: {task_type}, {score_to_output}: {f1 if task_type != 'qa' else precision}")


if __name__ == "__main__":
    # # 创建 ArgumentParser 对象
    # parser = argparse.ArgumentParser(description="Evaluate the prediction results based on the task type.")
    # # 添加任务类型参数
    # parser.add_argument("task_type", type=str, choices=['qa', 'tc', 're', 'ner'], help="Type of the task to evaluate")

    # # 解析命令行参数
    # args = parser.parse_args()

    

    # 调用主函数
    main(task_type, pred_resulf, log_file)

