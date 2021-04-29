# CCKS金融领域事件主体抽取
本次评测任务的文本范围包括互联网上的新闻文本，上市公司发布的公告文本。 
本次评测任务的事件类型包括：财务造假、偿付能力不足、高层失联/去世、企业破产、重大资损、重大赔付、重大事故、股权冻结、股权质押、增持、减持等。

该任务旨在从文本中抽取事件类型和对应的事件主体。即给定文本T，抽取T中所有的事件类型集合S，
对于S中的每个事件类型s，从文本T中抽取s的事件主体。其中各事件类型的主体实体类型为公司名称或人名或机构名称。 

输入： 一段文本T。<br>
输出：事件类型和事件主体。

# 技术日志
- [***2021/4/24  kevin***]<br>
  抽取预训练的词向量：sgns.financial.char(原始文件) ==> embedding_sgns_financial.npz<br>
  使用服创数据+爬取数据+ccks数据 [embeddingExtraction.py]


- [***2021/4/29  kevin***]<br>
  完成dataTool API 1.0版本<br>
  ClassificationDataSet ==> 返回文本token以及label<br>
  BIODataSet ==> 返回文本token以及对于的tag序列
