# 导入词表及配置
import pickle as pkl
from config import Config

vocab = pkl.load(open('embedding/vocab.pkl', 'rb'))
config = Config

# 导入label映射字典
label_dict = dict()

with open('data/label.txt', encoding='utf-8') as label_file:
    labels = label_file.readlines()

for label in labels:
    key, value = label.split('|')
    label_dict[key] = eval(value)

# 常量
UNK = '[UNK]'
PAD = '[PAD]'