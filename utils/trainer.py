from sklearn.metrics import f1_score, recall_score, precision_score, classification_report
import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_best = np.Inf
        self.delta = delta

    def __call__(self, val_score, model, save_path):

        score = val_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, model, save_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_score, model, save_path)
            self.counter = 0

    def save_checkpoint(self, val_score, model, save_path):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_score_best:.6f} --> {val_score:.6f}).  Saving model ...')
        torch.save(model.state_dict(), save_path)  # 存储最优模型的参数
        self.val_score_best = val_score


def train(train_loader, val_loader, net, optimizer, scheduler, config):
    # early stop初始化
    early_stopping = EarlyStopping(patience=config.patience, verbose=True)
    
    for epoch in range(config.epoch_size):
        for index, data in enumerate(train_loader):
            text, y, mask = data[0].to(config.device), data[1].to(config.device), data[2].to(config.device)

            mask = mask.bool()

            out, tag = net(text)
            
            crf_loss, predict_loss = net.module.loss_func(out, y, mask)
            loss = crf_loss + predict_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if index % 10 == 0:
                for i, data in enumerate(val_loader):
                    val_text, val_y = data[0].to(config.device), data[1]
                    out, tag = net(val_text)
                    val_f1 = evaluate(val_y, tag)
                print(f'GPU {config.local_rank} | epoch: {epoch + 1} batch: {index} '
                      f'| crf_loss: {crf_loss} predict_loss: {predict_loss} | val_f1: {val_f1}')
                
                if config.ifEarlyStop:
                    early_stopping(val_f1, net, config.save_path)
                    # 若满足early stopping要求则结束模型训练
                    if early_stopping.early_stop:
                        print("[EARLY-STOP] Stop at epoch-{} batch-{}".format(epoch + 1, index))
                        break
        else:
            # for循环正常执行则执行else中的语句，跳过外层break
            # scheduler.step()  # 更新学习率
            continue
            
        # for被break时执行外层break
        break

    return net


def evaluate(Y, Y_hat):
    f1 = 0
    num = len(Y)
    for y, y_hat in zip(Y, Y_hat):
        f1 += f1_score(y, y_hat, average='macro')

    return f1/num


def test(Y, Y_hat):
    y = []
    y_hat = []
    for i, j in zip(Y, Y_hat):
        y += list(i.numpy())
        y_hat += j
    
    report = classification_report(y, y_hat)
    print(report)