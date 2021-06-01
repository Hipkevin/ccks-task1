import torch

from utils import label_dict, vocab, UNK, PAD

from torch.utils.data import Dataset
from tqdm import tqdm


def word2idx(text: str, pad_size: int):
    """
    文本token化，并查询词表id
    :param pad_size: 填充长度
    :param text: 文本字符串
    :return: token id列表
    """
    token = list(text)
    mask = [1] * len(token)

    text_length = len(token)
    if text_length < pad_size:
        token.extend([PAD] * (pad_size - text_length))
        mask.extend([0] * (pad_size - text_length))
    else:
        token = token[0:pad_size]
        mask = mask[0:pad_size]

    words_id = list()
    for word in token:
        words_id.append(vocab.get(word, vocab.get(UNK)))

    return words_id, token, mask


def getTag(text, tag_content):
    """
    获取标注序列

    使用index字符串函数查找子串位置
    :param text: padding后的文本列表
    :param tag_content: 标注内容
    :return: 返回标注序列
    """
    text_length = len(text)
    tag_length = len(tag_content)

    tag = [0] * text_length
    text = ''.join(text)

    if tag_content in text:
        start = text.index(tag_content)
        tag[start] = 1
        tag[start + 1: start + tag_length] = [2] * (tag_length - 1)

    return tag


class ClassificationDataSet(Dataset):
    """
    分类数据集

    输入文件位置，返回torch数据集
    """

    def __getitem__(self, index):
        return self.text[index], self.label[index]

    def __init__(self, file_path, pad_size):
        super(ClassificationDataSet, self).__init__()

        text_list = list()
        label_list = list()

        # 读文件
        with open(file_path, encoding='utf-8') as file:
            sample = file.readlines()

        for s in tqdm(sample):
            # 取数据
            s = s.strip()
            s = s.split('|')
            text, label = s[0], label_dict[s[1]]

            # tokenize
            token, text = word2idx(text, pad_size)

            text_list.append(token)
            label_list.append(label)

        self.text = torch.LongTensor(text_list)
        self.label = torch.LongTensor(label_list)


class BIODataSet(Dataset):
    """
    标注数据集

    输入文件位置，返回torch数据集
    """

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        return self.text[index], self.tag[index], self.mask[index]

    def __init__(self, file_path, config):
        super(BIODataSet, self).__init__()
        
        pad_size = config.pad_size

        text_list = list()
        tag_list = list()
        mask_list = list()

        with open(file_path, encoding='utf-8') as file:
            sample = file.readlines()

        for s in tqdm(sample):
            s = s.strip()
            s = s.split('|')
            text, tagContent = s[0], s[1]

            token, text, mask = word2idx(text, pad_size)
            tag = getTag(text, tagContent)

            text_list.append(token)
            tag_list.append(tag)
            mask_list.append(mask)

        self.text = torch.LongTensor(text_list)
        self.tag = torch.LongTensor(tag_list)
        self.mask = torch.LongTensor(mask_list)


# """ 常量 """
# UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号
# TAG = {'公司舆情': 'CauseN',
#        '公司名称': 'CauseV',
#        }
#
# TAG_DICT = {"O": 0,
#             "B-Event": 1,
#             "I-Event": 2,
#
#             "B-Company": 3,
#             "I-Company": 4,
#
#             "<UNK>": 5,
#             "<PAD>": 6}
