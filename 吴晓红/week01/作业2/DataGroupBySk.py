import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
# 读取数据
data = pd.read_csv('dataset.csv',sep='\t',header=None)
# 向量化和训练
input_words = data[0].apply(lambda x: " " .join(jieba.lcut(x)))
vector = CountVectorizer()
vector.fit(input_words.values)
input_feature = vector.transform(input_words.values)
# 模型和训练
model = KNeighborsClassifier()
model.fit(input_feature,data[1].values)
print(model.classes_)
# 验证
test_prompt = ["从杭州到北京怎么走","给我播放建国大业的电影","前面红灯，你开车不要乱闯，部长在车上","拉肚子了"]
for prompt in test_prompt:
    test_words = " ".join(jieba.lcut(prompt))
    test_feature = vector.transform([test_words])
    rst = model.predict(test_feature)
    print("prompt " , prompt , "KNeighborsClassifier result ", rst)

