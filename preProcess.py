import re

if __name__ == '__main__':
    # 读原始数据
    with open('data/data.csv', encoding='utf-8') as file:
        data = file.readlines()

        save = []
        for d in data:
            # 清洗
            d = re.compile(u"[^\ta-zA-Z0-9\u4E00-\u9FA5]").sub('', d)

            sample = d.split('\t')

            # 过滤空值
            if sample[2] == 'NaN' or sample[3] == 'NaN' or len(sample) != 4:
                continue
            save.append(sample)

    # 加入分隔符并保存
    with open('data/data.txt', 'w', encoding='utf-8') as file:
        for s in save:
            sample = '|'.join(s[1:])
            file.write(sample+'\n')