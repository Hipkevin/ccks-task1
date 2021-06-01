import torch
import torch.nn as nn
import math
from torchcrf import CRF
import copy


class BiLstmCRF(nn.Module):
    def __init__(self, config):
        super(BiLstmCRF, self).__init__()

        # embedding随训练更新
        self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=True)

        self.device = config.device

        self.tag_num = config.tag_num
        self.seqLen = config.pad_size

        self.lstm = nn.LSTM(input_size=config.input_size,
                            hidden_size=config.hidden_size // 2,
                            num_layers=config.hidden_layer_num,
                            dropout=config.dropout,
                            bidirectional=True,  # 设置双向
                            batch_first=True)  # batch为张量的第一个维度

        self.output = nn.Linear(config.hidden_size // 2, config.tag_num)

        self.feedforward = FeedForwardNet(config.input_size, config.hidden_size // 2, 0.5, config.N)

        self.crf = CRF(config.tag_num, batch_first=True)

    def forward(self, x):
        # embedding层
        out = self.embedding(x)

        # 特征提取
        out, _ = self.lstm(out)
        out = self.feedforward(out)

        # 维度线性变换
        out = self.output(out)

        # crf解码
        # 计算loss时使用crf的前向传播
        tag = self.crf.decode(out)

        return out, tag

    def loss_func(self, out, y, mask):
        weight = torch.tensor([1] * self.tag_num, dtype=torch.float32).to(self.device)
        weight[0] = 0.08
        weight[1:] = 0.9

        criterion = nn.CrossEntropyLoss(weight=weight)

        crf_loss = - self.crf(out, y, mask, 'token_mean')  # crf-loss

        predict_loss = torch.tensor(0, dtype=torch.float32).to(self.device)
        for Y_hat, Y in zip(out, y):
            predict_loss += criterion(Y_hat, Y)

        return crf_loss, predict_loss


class FeedForwardNet(nn.Module):
    def __init__(self, input_size, output_size, dropout, N):
        super(FeedForwardNet, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.feed = self._clone(BasicFeedModule(input_size, dropout), N)
        self.output = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def _clone(self, module, N):
        return [module.clone().to(self.device) for _ in range(N)]

    def forward(self, x):
        res = x.clone()
        for feedforward in self.feed:
            x = feedforward(x)
        x += res

        out = self.output(x)
        out = self.dropout(out)

        return out


class BasicFeedModule(nn.Module):
    def __init__(self, n_dim, dropout):
        super(BasicFeedModule, self).__init__()
        self.fc1 = nn.Linear(n_dim, n_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(n_dim, n_dim)
        self.dropout = nn.Dropout(dropout)

    def clone(self):
        return copy.deepcopy(self)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.dropout(x)

        return x
