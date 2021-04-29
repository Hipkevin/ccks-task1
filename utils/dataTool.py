import torch

from utils import label_dict, vocab, UNK, PAD

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

def word2idx(text: str, pad_size: int):
    """
    文本token化，并查询词表id
    :param pad_size: 填充长度
    :param text: 文本字符串
    :return: token id列表
    """
    token = list(text)

    text_length = len(token)
    if text_length < pad_size:
        token += [PAD]*(text_length-pad_size)
    else:
        token = token[0:pad_size]

    words_id = list()
    for word in token:
        words_id.append(vocab.get(word, vocab.get(UNK)))

    return words_id

def getTag(text, tag_content):
    """
    获取标注序列

    使用index字符串函数查找子串位置
    :param text: padding后的文本
    :param tag_content: 标注内容
    :return: 返回标注序列
    """
    text_length = len(text)
    tag_length = len(tag_content)

    tag = [0]*text_length

    try:
        start = text.index(tag_content)
        tag[start] = 1
        tag[start+1: start+tag_length-1] = [2] * (tag_length - 1)
    except ValueError:
        return tag

    return tag

class ClassificationDataSet(Dataset):
    """
    分类数据集

    输入文件位置，返回torch数据集
    """
    def __getitem__(self, index) -> T_co:
        return self.text[index], self.label[index]

    def __init__(self, file_path, pad_size):
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
            text = word2idx(text, pad_size)

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

    def __init__(self, file_path, pad_size):
        super(BIODataSet, self).__init__()

        text_list = list()
        tag_list = list()

        with open(file_path, encoding='utf-8') as file:
            sample = file.readlines()

        for s in sample:
            s = s.split('|')
            text, tagContent = s[0], s[1]

            text = word2idx(text, pad_size)
            tag = getTag(text, tagContent)

            text_list.append(text)
            tag_list.append(tag)

        self.text = torch.LongTensor(text_list)
        self.tag = torch.LongTensor(tag_list)