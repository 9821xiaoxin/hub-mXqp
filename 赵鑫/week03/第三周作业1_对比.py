import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import time
# 自定义数据集 - 》 为每个任务定义单独的数据集的读取方式，这个任务的输入和输出
# 统一的写法，底层pytorch 深度学习 / 大模型
class CharLSTMDataset(Dataset):
    # 初始化
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts # 文本输入
        self.labels = torch.tensor(labels, dtype=torch.long) # 文本对应的标签
        self.char_to_index = char_to_index # 字符到索引的映射关系
        self.max_len = max_len # 文本最大输入长度

    # 返回数据集样本个数
    def __len__(self):
        return len(self.texts)

    # 获取当个样本
    def __getitem__(self, idx):
        text = self.texts[idx]
        # pad and crop
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]

# a = CharLSTMDataset()
# len(a) -> a.__len__
# a[0] -> a.__getitem__


# --- NEW LSTM Model Class ---
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()

        # 词表大小 转换后维度的维度
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # 随机编码的过程， 可训练的
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)  # 循环层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # batch size * seq length -》 batch size * seq length * embedding_dim
        embedded = self.embedding(x)

        # batch size * seq length * embedding_dim -》 batch size * seq length * hidden_dim
        lstm_out, (hidden_state, cell_state) = self.lstm(embedded)

        # batch size * output_dim
        out = self.fc(hidden_state.squeeze(0))
        return out
# # --- Training and Prediction ---
# lstm_dataset = CharLSTMDataset(texts, numerical_labels, char_to_index, max_len)
# lstm_dataloader = DataLoader(lstm_dataset, batch_size=32, shuffle=True)
#
# embedding_dim = 64
# hidden_dim = 128
# output_dim = len(label_to_index)
#
# lstm_model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
# lstm_criterion = nn.CrossEntropyLoss()
# lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

# 替换为gru
# class CharGRUDataset(Dataset):
#     """
#     字符级GRU数据集类
#     用于将文本数据转换为模型可训练的张量格式
#     """
#     def __init__(self, texts, labels, char_to_index, max_len):
#         self.texts = texts  # 文本输入列表
#         self.labels = torch.tensor(labels, dtype=torch.long)  # 文本对应标签，转换为long类型张量
#         self.char_to_index = char_to_index  # 字符到索引的映射字典
#         self.max_len = max_len  # 文本最大输入长度，用于序列填充或截断
#
#     def __len__(self):
#         """返回数据集中样本的总数"""
#         return len(self.texts)
#
#     def __getitem__(self, idx):
#         """
#         获取单个样本的数据和标签
#         Args:
#             idx: 样本索引
#         Returns:
#             indices_tensor: 编码后的文本张量
#             label: 对应的标签
#         """
#         text = self.texts[idx]
#         # 将文本转换为索引序列，超出max_len的部分截断
#         indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
#         # 序列长度不足max_len时进行填充（padding）
#         indices += [0] * (self.max_len - len(indices))
#         # 转换为PyTorch张量
#         indices_tensor = torch.tensor(indices, dtype=torch.long)
#         return indices_tensor, self.labels[idx]


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

# --- 训练和预测部分 ---

# # 创建GRU数据集实例
# # 参数说明:
# # - texts: 原始文本数据列表
# # - numerical_labels: 数值化后的标签列表
# - char_to_index: 字符到索引的映射字典
# # - max_len: 文本序列的最大长度
# gru_dataset = CharGRUDataset(texts, numerical_labels, char_to_index, max_len)
#
# # 创建数据加载器
# # batch_size=32: 每个批次包含32个样本
# # shuffle=True: 每个epoch随机打乱数据顺序
# gru_dataloader = DataLoader(gru_dataset, batch_size=32, shuffle=True)
#
# # 设置模型超参数
# embedding_dim = 64    # 词嵌入维度，决定每个字符的向量表示大小
# hidden_dim = 128      # GRU隐藏层维度，影响模型的记忆能力
# output_dim = len(label_to_index)  # 输出维度等于类别数量
#
# # 初始化GRU分类模型
# # 参数传递词汇表大小、嵌入维度、隐藏层维度和输出维度
# gru_model = GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
#
# # 定义损失函数
# # CrossEntropyLoss适用于多分类任务，内部包含softmax操作
# gru_criterion = nn.CrossEntropyLoss()
#
# # 定义优化器
# # Adam优化器，学习率设置为0.001
# # model.parameters()传入模型所有可训练参数
# gru_optimizer = optim.Adam(gru_model.parameters(), lr=0.001)

# class CharRNNDataset(Dataset):
#     """
#     字符级RNN数据集类
#     用于将文本数据转换为模型可训练的张量格式
#     """
#
#     def __init__(self, texts, labels, char_to_index, max_len):
#         self.texts = texts  # 文本输入列表
#         self.labels = torch.tensor(labels, dtype=torch.long)  # 文本对应标签，转换为long类型张量
#         self.char_to_index = char_to_index  # 字符到索引的映射字典
#         self.max_len = max_len  # 文本最大输入长度，用于序列填充或截断
#
#     def __len__(self):
#         """返回数据集中样本的总数"""
#         return len(self.texts)
#
#     def __getitem__(self, idx):
#         """
#         获取单个样本的数据和标签
#         Args:
#             idx: 样本索引
#         Returns:
#             indices_tensor: 编码后的文本张量
#             label: 对应的标签
#         """
#         text = self.texts[idx]
#         # 将文本转换为索引序列，超出max_len的部分截断
#         indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
#         # 序列长度不足max_len时进行填充（padding）
#         indices += [0] * (self.max_len - len(indices))
#         # 转换为PyTorch张量
#         indices_tensor = torch.tensor(indices, dtype=torch.long)
#         return indices_tensor, self.labels[idx]


class RNNClassifier(nn.Module):
    """
    基于RNN的文本分类模型
    使用嵌入层+RNN循环层+全连接层的结构
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        """
        初始化模型各层
        Args:
            vocab_size: 词汇表大小
            embedding_dim: 词嵌入维度
            hidden_dim: RNN隐藏层维度
            output_dim: 输出类别数
        """
        super(RNNClassifier, self).__init__()

        # 词嵌入层：将稀疏的one-hot向量转换为稠密的embedding向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # RNN循环层：处理序列数据，batch_first=True表示输入形状为(batch, seq, feature)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)

        # 全连接层：将RNN输出映射到分类结果
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

        # RNN处理：embedded -> rnn_output:(batch_size, seq_len, hidden_dim)
        #          hidden_state:(num_layers*num_directions, batch_size, hidden_dim)
        rnn_out, hidden_state = self.rnn(embedded)

        # 使用最后一个时间步的隐藏状态进行分类
        # hidden_state.squeeze(0): 去除第0维的单维度，得到(batch_size, hidden_dim)
        out = self.fc(hidden_state.squeeze(0))
        return out

# # --- 训练和预测部分 ---
# # 创建GRU数据集实例
# # 参数说明:
# # - texts: 原始文本数据列表
# # - numerical_labels: 数值化后的标签列表
# # - char_to_index: 字符到索引的映射字典
# # - max_len: 文本序列的最大长度
# rnn_dataset = CharRNNDataset(texts, numerical_labels, char_to_index, max_len)
#
# # 创建数据加载器
# # batch_size=32: 每个批次包含32个样本
# # shuffle=True: 每个epoch随机打乱数据顺序
# rnn_dataloader = DataLoader(rnn_dataset, batch_size=32, shuffle=True)
#
# # 设置模型超参数
# embedding_dim = 64    # 词嵌入维度，决定每个字符的向量表示大小
# hidden_dim = 128      # GRU隐藏层维度，影响模型的记忆能力
# output_dim = len(label_to_index)  # 输出维度等于类别数量
#
# # 初始化GRU分类模型
# # 参数传递词汇表大小、嵌入维度、隐藏层维度和输出维度
# rnn_model = RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
#
# # 定义损失函数
# # CrossEntropyLoss适用于多分类任务，内部包含softmax操作
# rnn_criterion = nn.CrossEntropyLoss()
#
# # 定义优化器
# # Adam优化器，学习率设置为0.001
# # model.parameters()传入模型所有可训练参数
# rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=0.001)

# num_epochs = 4
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     for idx, (inputs, labels) in enumerate(gru_dataloader):
#         gru_optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         gru_optimizer.step()
#         running_loss += loss.item()
#         if idx % 50 == 0:
#             print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")
#
#     print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(gru_dataloader):.4f}")

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

# index_to_label = {i: label for label, i in label_to_index.items()}
#
# new_text = "帮我导航到北京"
# predicted_class = classify_text_lstm(new_text, model, char_to_index, max_len, index_to_label)
# print(f"输入 '{new_text}' 预测为: '{predicted_class}'")
#
# new_text_2 = "查询明天北京的天气"
# predicted_class_2 = classify_text_lstm(new_text_2, model, char_to_index, max_len, index_to_label)
# print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")


def train_model(model, dataloader, criterion, optimizer, device, epochs=5):
    model.train()
    accuracies = []

    for epoch in range(epochs):
        correct = 0
        total = 0
        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        epoch_acc = 100 * correct / total
        accuracies.append(epoch_acc)

    return accuracies


def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    accuracy = accuracy_score(all_targets, all_preds)
    return accuracy

def main():
    print("开始LSTM、GRU、RNN模型对比实验...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 数据加载
    dataset = pd.read_csv("../data/dataset.csv", sep="\t", header=None)
    # 获取第一列作为文本数据
    texts = dataset[0].tolist()
    print(f"数据集大小: {len(texts)} 条文本")
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
    embedding_dim = 64  # 词嵌入维度，决定每个字符的向量表示大小
    hidden_dim = 128  # GRU隐藏层维度，影响模型的记忆能力
    output_dim = len(label_to_index)  # 输出维度等于类别数量

    dataset = CharLSTMDataset(texts, numerical_labels, char_to_index, max_len)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    models = {
        'LSTM': LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim),
        'GRU': GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim),
        'RNN': RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
    }

    criterion = nn.CrossEntropyLoss()
    results = {}

    for name, model in models.items():
        print(f"\n训练{name}模型...")
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        start_time = time.time()
        accuracies = train_model(model, dataloader, criterion, optimizer, device, epochs=10)
        training_time = time.time() - start_time

        print(f"评估{name}模型...")
        final_accuracy = evaluate_model(model, dataloader, device)

        results[name] = {
            'accuracies': accuracies,
            'final_acc': final_accuracy,
            'time': training_time
        }

        print(f"{name}模型最终准确率: {final_accuracy:.2f}%")
        print(f"{name}模型训练时间: {training_time:.2f} 秒")

    print(f"\n模型对比结果汇总:")
    print(f"{'模型':<10} {'最终准确率':<12} {'训练时间':<12}")
    print("-" * 35)
    for name, data in results.items():
        print(f"{name:<10} {data['final_acc']:<12.2f} {data['time']:<12.2f}")

    best_model = max(results.items(), key=lambda x: x[1]['final_acc'])
    print(f"\n最佳模型: {best_model[0]} (准确率: {best_model[1]['final_acc']:.2f}%)")

if __name__ == "__main__":
    main()

# 运行结果
开始LSTM、GRU、RNN模型对比实验...
使用设备: cpu
数据集大小: 12100 条文本

训练LSTM模型...
评估LSTM模型...
LSTM模型最终准确率: 0.98%
LSTM模型训练时间: 218.82 秒

训练GRU模型...
评估GRU模型...
GRU模型最终准确率: 1.00%
GRU模型训练时间: 178.84 秒

训练RNN模型...
评估RNN模型...
RNN模型最终准确率: 0.12%
RNN模型训练时间: 105.99 秒

模型对比结果汇总:
模型         最终准确率        训练时间        
-----------------------------------
LSTM       0.98         218.82      
GRU        1.00         178.84      
RNN        0.12         105.99      

最佳模型: GRU (准确率: 1.00%)
