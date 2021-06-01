import torch
import numpy as np


class Config:
    def __init__(self):
        super(Config, self).__init__()
        
        # 设备管理
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.local_rank = -1
        
        # embedding加载
        self.embedding_pretrained = torch.tensor(
            np.load('embedding/embedding_sgns_financial.npz')['embeddings'].astype('float32')
        )
        
        # 模型保存路径
        self.save_path = 'model/BiLstmCRF_test.pt'
        
        # 填充长度
        self.pad_size = 128
        
        # LSTM维度参数
        self.input_size = 300
        self.hidden_size = 300
        self.dropout = 0.5
        
        # 全连接层数及LSTM隐藏层数量
        self.hidden_layer_num = 5
        self.N = 5
        
        # 标签数
        self.tag_num = 3
        
        # 训练参数
        self.learning_rate = 1e-3
        self.decay = 1e-4
        self.batch_size = 256
        self.epoch_size = 60
        
        self.T_0 = 5
        self.T_mult = 2
        
        self.ifEarlyStop = False
        self.patience = 10