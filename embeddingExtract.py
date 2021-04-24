import numpy as np
import pickle as pkl
from tqdm import tqdm

MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '[UNK]', '[PAD]'  # 未知字，padding符号


def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


if __name__ == '__main__':
    '''
    提取预训练词向量
    
    只需要修改数据目录
    '''
    train_dir = "embedding/corpus.txt"
    vocab_dir = "embedding/vocab.pkl"
    pretrain_dir = "embedding/sgns.financial.char"
    filename_trimmed_dir = "embedding/embedding_sgns_financial"  # 原始词向量文件
    emb_dim = 300

    tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
    word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
    pkl.dump(word_to_id, open(vocab_dir, 'wb'))

    embeddings = np.random.rand(len(word_to_id), emb_dim)
    f = open(pretrain_dir, "r", encoding='UTF-8')
    for i, line in enumerate(f.readlines()):
        lin = line.strip().split(" ")
        if lin[0] in word_to_id:
            idx = word_to_id[lin[0]]
            emb = [float(x) for x in lin[1:301]]
            embeddings[idx] = np.asarray(emb, dtype='float32')
    f.close()
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)