from utils.dataTool import ClassificationDataSet, BIODataSet
from config import Config

if __name__ == '__main__':
    config = Config()

    # cDataSet = ClassificationDataSet('data/classification_data.txt', config.pad_size)
    bDataSet = BIODataSet('data/bio_data.txt', config.pad_size)

    for idx, data in enumerate(bDataSet):
        text, label = data[0], data[1]

        print(text, label)