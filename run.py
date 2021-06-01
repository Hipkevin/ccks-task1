import time

import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from config import Config
from utils.dataTool import BIODataSet
from utils.model import BiLstmCRF
from utils.trainer import train, test

# DDP后端初始化
torch.distributed.init_process_group(backend="nccl")


if __name__ == '__main__':
    config = Config()
    
    # 设置每个进程中GPU的local_rank
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    config.device = torch.device("cuda", local_rank)
    config.local_rank = local_rank
    
    # 模型初始化及DDP封装
    net = BiLstmCRF(config).to(config.device)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank])
    
    print(f'GPU {local_rank}', 'Loading...')
    train_data = BIODataSet('data/train_3w.txt', config)
    val_data = list(BIODataSet('data/val_3w.txt', config))
    test_data = list(BIODataSet('data/test_3w.txt', config))
    
    # train和val数据DDP封装
    loader = DataLoader(train_data, batch_size=config.batch_size,
                        pin_memory=True, num_workers=8, drop_last=True, sampler=DistributedSampler(train_data))
    val_loader = DataLoader(val_data, batch_size=200, sampler=DistributedSampler(val_data))
    
    # test在GPU-0计算
    test_loader = DataLoader(test_data, batch_size=200)
    
    # from torch.optim import RMSprop, SGD, Adam
    # optimizer = torch.optim.Adam(params=net.parameters(), lr=config.learning_rate, weight_decay=config.decay)
    # optimizer = torch.optim.RMSprop(params=net.parameters(), lr=config.learning_rate, weight_decay=config.decay)
    optimizer = torch.optim.SGD(params=net.parameters(), lr=config.learning_rate, weight_decay=config.decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config.T_0, T_mult=config.T_mult, eta_min=1e-8)
    
    print(f'GPU {local_rank}', 'Training...')
    tik = time.time()
    net = train(train_loader=loader,
                val_loader=val_loader,
                net=net,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config)
    tok = time.time()
    print(f'GPU {local_rank}', 'Training Time: ', tok-tik)
    
    # GPU-0评估保存模型
    if local_rank == 0:
        Y = []
        Tag = []
        for i, data in enumerate(test_loader):
            text, y = data[0].to(config.device), data[1]
            out, tag = net(text)
            
            Y += y
            Tag += tag
        test(Y, Tag)

        torch.save(net.module.state_dict(), 'model/BiLstmCRF_large_DDP_200ep.pt')
