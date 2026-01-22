import pandas as pd
from openai import OpenAI
# 读取数据
data = pd.read_csv('dataset.csv',sep='\t',header=None)
unique_values = data[1].unique()

question = "从杭州到北京怎么走"
unique_values_str = ','.join(map(str, unique_values))
template = "将文本 %s 进行分类，类别为以下内容: " + unique_values_str
prompt = template % question
print(prompt)

client = OpenAI(
    api_key="sk-xxx",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
completion = client.chat.completions.create(
    model="qwen-plus",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."}, # 给大模型的命令，角色的定义
        {"role": "user", "content": prompt},  # 用户的提问
    ]
)
rst = completion.choices[0].message.content
print("prompt ",question, "  result ",  rst)