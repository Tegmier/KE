from transformers import BertTokenizer, BertModel
import torch

# 加载预训练模型和tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 准备输入文本
input_text1 = "Hugging Face is creating a tool that democratizes AI."
input_text2 = "I love to use Hugging Face."

inputs1 = tokenizer(input_text1, return_tensors="pt")
inputs2 = tokenizer(input_text2, return_tensors="pt")

with torch.no_grad():
    outputs1 = model(**inputs1)
    outputs2 = model(**inputs2)

tokens1 = tokenizer.convert_ids_to_tokens(inputs1['input_ids'][0])
tokens2 = tokenizer.convert_ids_to_tokens(inputs2['input_ids'][0])

last_hidden_states1 = outputs1.last_hidden_state
last_hidden_states2 = outputs2.last_hidden_state

for token,embedding in zip(tokens1, last_hidden_states1[0]):
    print(f"Token: {token}")
    print(f"Embedding: {embedding[:5]}...")  

for token,embedding in zip(tokens2, last_hidden_states2[0]):
    print(f"Token: {token}")
    print(f"Embedding: {embedding[:5]}...") 



# # 使用tokenizer将文本编码为token IDs
# inputs = tokenizer(input_text, return_tensors="pt")

# # 在模型中前向传播输入
# with torch.no_grad():
#     outputs = model(**inputs)

# # 获取所有层的输出
# last_hidden_states = outputs.last_hidden_state

# # 打印最后一层的输出
# print(last_hidden_states)

# # 获取tokenized后的单词列表
# tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

# # 打印每个单词的嵌入
# for token, embedding in zip(tokens, last_hidden_states[0]):
#     print(f"Token: {token}")
#     print(f"Embedding: {embedding[:5]}...")  # 只显示前5个值
#     print(embedding.shape)
#     print()
