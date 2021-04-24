import torch

from utils import label_dict, vocab, UNK, PAD

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

def word2idx(text: str):
    """
    文本token化，并查询词表id
    :param text: 文本字符串
    :return: token id列表
    """
    token = list(text)

    words_id = list()
    for word in token:
        words_id.append(vocab.get(word, vocab.get(UNK)))

    return words_id

class ClassificationDataSet(Dataset):
    """
    分类数据集

    输入文件位置，返回torch数据集
    """
    def __getitem__(self, index) -> T_co:
        return self.text[index], self.label[index]

    def __init__(self, file_path):
        super(ClassificationDataSet, self).__init__()

        text_list = list()
        label_list = list()

        # 读文件
        with open(file_path, encoding='utf-8') as file:
            sample = file.readlines()

        for s in sample:
            # 取数据
            s = s.split('|')
            text, label = s[0], label_dict[s[1]]

            # tokenize
            text = word2idx(text)

            text_list.append(text)
            label_list.append(label)

        self.text = torch.LongTensor(text_list)
        self.label = torch.LongTensor(label_list)

class BIODataSet(Dataset):
    """
    标注数据集

    输入文件位置，返回torch数据集
    """
    def __getitem__(self, index) -> T_co:
        return self.text[index], self.tag[index]

    def __init__(self, file_path):
        super(BIODataSet, self).__init__()

        text_list = list()
        tag_list = list()

        with open(file_path, encoding='utf-8') as file:
            sample = file.readlines()

        for s in sample:
            s = s.split('|')
            text, tag = s[0], label_dict[s[1]]

            text = word2idx(text)

            text_list.append(text)
            tag_list.append(tag)

        self.text = torch.LongTensor(text_list)
        self.tag = torch.LongTensor(tag_list)