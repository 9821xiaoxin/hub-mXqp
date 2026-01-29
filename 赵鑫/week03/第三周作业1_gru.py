import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 数据加载
dataset = pd.read_csv("../data/dataset.csv", sep="\t", header=None)
# 获取第一列作为文本数据
texts = dataset[0].tolist()
# 获取第二列作为字符串标签
string_labels = dataset[1].tolist()
# 创建标签到索引的映射字典
# 使用set去除重复标签，然后为每个唯一标签分配一个整数索引
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
# 将字符串标签转换为数值标签
numerical_labels = [label_to_index[label] for label in string_labels]

# 创建字符到索引的映射字典
# 首先添加特殊标记<pad>，索引为0
char_to_index = {'<pad>': 0}
# 遍历所有文本，构建字符词汇表
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

# 创建索引到字符的映射字典，用于后续解码
index_to_char = {i: char for char, i in char_to_index.items()}
# 获取词汇表大小
vocab_size = len(char_to_index)

# max length 最大输入的文本长度
max_len = 40

# 自定义数据集 - 》 为每个任务定义单独的数据集的读取方式，这个任务的输入和输出
# 统一的写法，底层pytorch 深度学习 / 大模型
# class CharLSTMDataset(Dataset):
#     # 初始化
#     def __init__(self, texts, labels, char_to_index, max_len):
#         self.texts = texts # 文本输入
#         self.labels = torch.tensor(labels, dtype=torch.long) # 文本对应的标签
#         self.char_to_index = char_to_index # 字符到索引的映射关系
#         self.max_len = max_len # 文本最大输入长度
#
#     # 返回数据集样本个数
#     def __len__(self):
#         return len(self.texts)
#
#     # 获取当个样本
#     def __getitem__(self, idx):
#         text = self.texts[idx]
#         # pad and crop
#         indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
#         indices += [0] * (self.max_len - len(indices))
#         return torch.tensor(indices, dtype=torch.long), self.labels[idx]

# a = CharLSTMDataset()
# len(a) -> a.__len__
# a[0] -> a.__getitem__


# --- NEW LSTM Model Class ---
# class LSTMClassifier(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
#         super(LSTMClassifier, self).__init__()
#
#         # 词表大小 转换后维度的维度
#         self.embedding = nn.Embedding(vocab_size, embedding_dim) # 随机编码的过程， 可训练的
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)  # 循环层
#         self.fc = nn.Linear(hidden_dim, output_dim)
#
#     def forward(self, x):
#         # batch size * seq length -》 batch size * seq length * embedding_dim
#         embedded = self.embedding(x)
#
#         # batch size * seq length * embedding_dim -》 batch size * seq length * hidden_dim
#         lstm_out, (hidden_state, cell_state) = self.lstm(embedded)
#
#         # batch size * output_dim
#         out = self.fc(hidden_state.squeeze(0))
#         return out
# --- Training and Prediction ---
# lstm_dataset = CharLSTMDataset(texts, numerical_labels, char_to_index, max_len)
# dataloader = DataLoader(lstm_dataset, batch_size=32, shuffle=True)
#
# embedding_dim = 64
# hidden_dim = 128
# output_dim = len(label_to_index)
#
# model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# 替换为gru
class CharGRUDataset(Dataset):
    """
    字符级GRU数据集类
    用于将文本数据转换为模型可训练的张量格式
    """
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts  # 文本输入列表
        self.labels = torch.tensor(labels, dtype=torch.long)  # 文本对应标签，转换为long类型张量
        self.char_to_index = char_to_index  # 字符到索引的映射字典
        self.max_len = max_len  # 文本最大输入长度，用于序列填充或截断

    def __len__(self):
        """返回数据集中样本的总数"""
        return len(self.texts)

    def __getitem__(self, idx):
        """
        获取单个样本的数据和标签
        Args:
            idx: 样本索引
        Returns:
            indices_tensor: 编码后的文本张量
            label: 对应的标签
        """
        text = self.texts[idx]
        # 将文本转换为索引序列，超出max_len的部分截断
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        # 序列长度不足max_len时进行填充（padding）
        indices += [0] * (self.max_len - len(indices))
        # 转换为PyTorch张量
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        return indices_tensor, self.labels[idx]


class GRUClassifier(nn.Module):
    """
    基于GRU的文本分类模型
    使用嵌入层+GRU循环层+全连接层的结构
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        """
        初始化模型各层
        Args:
            vocab_size: 词汇表大小
            embedding_dim: 词嵌入维度
            hidden_dim: GRU隐藏层维度
            output_dim: 输出类别数
        """
        super(GRUClassifier, self).__init__()

        # 词嵌入层：将稀疏的one-hot向量转换为稠密的embedding向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # GRU循环层：处理序列数据，batch_first=True表示输入形状为(batch, seq, feature)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

        # 全连接层：将GRU输出映射到分类结果
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        前向传播过程
        Args:
            x: 输入张量，形状为(batch_size, sequence_length)
        Returns:
            out: 分类结果，形状为(batch_size, output_dim)
        """
        # 词嵌入：(batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(x)

        # GRU处理：embedded -> gru_output:(batch_size, seq_len, hidden_dim)
        #          hidden_state:(num_layers*num_directions, batch_size, hidden_dim)
        gru_out, hidden_state = self.gru(embedded)

        # 使用最后一个时间步的隐藏状态进行分类
        # hidden_state.squeeze(0): 去除第0维的单维度，得到(batch_size, hidden_dim)
        out = self.fc(hidden_state.squeeze(0))
        return out

# --- Training and Prediction ---
lstm_dataset = CharGRUDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(lstm_dataset, batch_size=32, shuffle=True)

embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)

model = GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 4
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if idx % 50 == 0:
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

def classify_text_lstm(text, model, char_to_index, max_len, index_to_label):
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label

index_to_label = {i: label for label, i in label_to_index.items()}

new_text = "帮我导航到北京"
predicted_class = classify_text_lstm(new_text, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text_lstm(new_text_2, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")

# 运行结果
Batch 个数 0, 当前Batch Loss: 2.49912166595459
Batch 个数 50, 当前Batch Loss: 2.5192880630493164
Batch 个数 100, 当前Batch Loss: 2.314656972885132
Batch 个数 150, 当前Batch Loss: 1.655076265335083
Batch 个数 200, 当前Batch Loss: 1.091538906097412
Batch 个数 250, 当前Batch Loss: 0.7952993512153625
Batch 个数 300, 当前Batch Loss: 0.611301064491272
Batch 个数 350, 当前Batch Loss: 0.6169086694717407
Epoch [1/4], Loss: 1.3998
Batch 个数 0, 当前Batch Loss: 0.6831258535385132
Batch 个数 50, 当前Batch Loss: 0.39044705033302307
Batch 个数 100, 当前Batch Loss: 0.6044697165489197
Batch 个数 150, 当前Batch Loss: 0.7187166810035706
Batch 个数 200, 当前Batch Loss: 0.12334746867418289
Batch 个数 250, 当前Batch Loss: 0.6446067094802856
Batch 个数 300, 当前Batch Loss: 0.18769295513629913
Batch 个数 350, 当前Batch Loss: 0.3538445830345154
Epoch [2/4], Loss: 0.4547
Batch 个数 0, 当前Batch Loss: 0.3337913751602173
Batch 个数 50, 当前Batch Loss: 0.14884865283966064
Batch 个数 100, 当前Batch Loss: 0.2314063012599945
Batch 个数 150, 当前Batch Loss: 0.3167068064212799
Batch 个数 200, 当前Batch Loss: 0.14145313203334808
Batch 个数 250, 当前Batch Loss: 0.6335572004318237
Batch 个数 300, 当前Batch Loss: 0.04653419554233551
Batch 个数 350, 当前Batch Loss: 0.3625326454639435
Epoch [3/4], Loss: 0.3030
Batch 个数 0, 当前Batch Loss: 0.15402768552303314
Batch 个数 50, 当前Batch Loss: 0.20025727152824402
Batch 个数 100, 当前Batch Loss: 0.19655057787895203
Batch 个数 150, 当前Batch Loss: 0.20549416542053223
Batch 个数 200, 当前Batch Loss: 0.13411934673786163
Batch 个数 250, 当前Batch Loss: 0.11179609596729279
Batch 个数 300, 当前Batch Loss: 0.11357773840427399
Batch 个数 350, 当前Batch Loss: 0.1477225422859192
Epoch [4/4], Loss: 0.2200
输入 '帮我导航到北京' 预测为: 'Travel-Query'
输入 '查询明天北京的天气' 预测为: 'Weather-Query'

