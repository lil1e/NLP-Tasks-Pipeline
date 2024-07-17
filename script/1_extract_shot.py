
from transformers import BertModel, BertTokenizer
import torch
import json
from tqdm import tqdm
import faiss
import numpy as np
import os
import yaml

#-------------------------------配置信息------------------------------------

#配置config.yaml文件路径
config_file_path = '/home/zjy/14_project/config.yaml'

#------------------------------以下无需更改----------------------------------
def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config(config_file_path)['1_extract_shot']

# 输入输出
procesd_set = config['procesd_set']
extractd_set_temp = config['extractd_set']

#embedding模型
model_name = config['model_name']

#shot数
shot_num = config['shot_num']

#GPU
device_map = config['device_map']

# 构建新的文件名，通过在.json前插入'_{shot_num}'
extractd_set = extractd_set_temp.replace('.json', f'_{shot_num}shot.json')

#----------------------------加载数据生成嵌入-----------------------------

# 设置环境变量以避免内存碎片
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


# 加载数据集
with open(procesd_set, encoding='utf-8') as f:
    dataset = json.load(f)


# 加载模型和分词器
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# 将模型移到 GPU
device = torch.device(device_map if torch.cuda.is_available() else 'cpu')
model.to(device)

# 生成嵌入
def generate_embeddings(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    # 将输入张量移到 GPU
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():  # 禁用梯度计算，减少内存占用
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # 平均池化获取句子嵌入
    return embeddings

# 嵌入列表
embeddings_list = []

batch_size = 32  # 批处理大小，根据需要调整
#————————————————————————————————————————————————————————————————————————这个函数可能需要改
for i in tqdm(range(0, len(dataset), batch_size), desc="Generating embeddings"):
    batch_data = dataset[i:i+batch_size]
    for data in batch_data:
        query_text = data['conversation'][0]["human"]
        embeddings = generate_embeddings(query_text)
        embeddings_list.append(embeddings)
        # break

# 清理缓存
torch.cuda.empty_cache()

#------------------------------构建检索库-------------------------

#因为numpy数组要在cpu上处理，所以由两个列表
embedding_np_list = []

for embeddings in tqdm(embeddings_list, desc="Building index"):
    # 将嵌入转换为 numpy 数组
    embeddings_np = embeddings.detach().cpu().numpy()  # 直接转换为 numpy 数组
    # 对嵌入进行 L2 归一化
    embeddings_np = embeddings_np / np.linalg.norm(embeddings_np, axis=1, keepdims=True)
    embedding_np_list.append(embeddings_np)

# 将所有嵌入合并为一个 numpy 数组
all_embeddings_np = np.vstack(embedding_np_list)

# 构建 Faiss 索引（基于内积）————归一化的向量内积就是余弦相似度
dimension = embeddings_np.shape[1]
index = faiss.IndexFlatIP(dimension)

# # 如果使用 GPU，添加如下代码——！显存会不足
# res = faiss.StandardGpuResources()
# index = faiss.index_cpu_to_gpu(res, 0, index)

# 添加嵌入到索引中
index.add(all_embeddings_np)

#-------------------------------------相似度检索------------------------------

# 定义检索函数
def retrieve_top_k_similar(query_text, k=shot_num):
    # 生成查询文本的嵌入向量
    query_embeddings = generate_embeddings(query_text)
    query_embeddings_np = query_embeddings.detach().cpu().numpy()  # 直接转换为 numpy 数组
    # 对查询向量进行 L2 归一化
    query_embeddings_np = query_embeddings_np / np.linalg.norm(query_embeddings_np)
    query_embeddings_np = query_embeddings_np / np.linalg.norm(query_embeddings_np)
    # 使用 Faiss 检索最相似的向量
    distances, indices = index.search(query_embeddings_np, k+1)  # 检索包括自身在内的最相似的 k+1 个向量
    # 构建相似列表
    similar_list = []
    
    for i in range(1, k+1):  # 跳过第一个，因为它是自身
        # 提取数据集的索引
        data_index = indices[0][i]
        # 访问数据集中的对话内容
        conversation_data = dataset[data_index]['conversation'][0]  
       # 使用字典推导式创建新的相似项字典
        similar_item = {
            "conversation_id": dataset[data_index]['conversation_id'],
            "human": conversation_data['human'],
            "assistant": conversation_data['assistant']
        }
       
        similar_list.append(similar_item)
    return similar_list

#---------------------------------生成extractd_set----------------------------------

json_data = []
assistant = []

for data in tqdm(dataset, desc="Building extractd_set"):
    human = data['conversation'][0]["human"]
    assistant = data['conversation'][0]['assistant']
    conversation_id = data['conversation_id']
    # 获取相似列表
    similar_list = retrieve_top_k_similar(human, k=shot_num)
    # 构建新的数据元素
    new_item = {
        "conversation_id": conversation_id,
        "human": human,
        "example": similar_list,
        "assistant": assistant
    }
    json_data.append(new_item)
    # break

# 保存新数据集为 JSON 文件
with open(extractd_set, 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=4)

print(f"extractd_set saved to {extractd_set}")


# '''