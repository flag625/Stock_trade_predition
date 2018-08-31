# Stock_trade_predition
本项目是研究性质的，对他人的预测模型进行研究学习，不用于商用目的。

文件夹：
1、LSTM_LOSS_MODEL
项目：利用LSTM网络和风险估计损失函数做股票交易
出处：https://github.com/happynoom/DeepTrade
内容摘要：
   本项目的网络结构如下：
     LSTM layer  ->  dense connected layer  ->  batch normalization layer  -> activation layer
     激活函数：relu6
   损失函数：
       损失函数定义为： Loss = -100. * mean(P * (R-c))
       P为网络输出的集合，也就是持仓策略的集合（在0和1之间取值），
       R为相应的第二天的价格变化率的集合,
       c为投资成本。

2、LSTM_demo
对项目的LSTM模型框架进行梳理，整理出demo，简述其框架。
以及test.py测试数据在tensorflow计算图中的shape变化。

#复现代码
3、CommonAPI
项目复现脚本的常用函数与配置文件。
A、log文件夹：数据库日志，记录数据在获取、存储、提取时出现的异常。
B、DBcommom.py：数据库常用操作脚本，包括数据库连接、数据提取等操作。
C、base.py：常用操作脚本，包括数据库连接选择、数据存储等操作。
D、threadAPI.py：自定义多线程类。用于辅助提升查询与存储大量数据的效率。
E、test.conf：配置文件，包括数据库连接配置、数据库日志存储路径。

4、LSTM_LOSS_test
LSTM模型复现，并根据本地需求作出调整，包括：MySQL存储原始数据，模型能够迭代训练多个股票的模型。
A、rawdata.py：
    建立RawData类，stockhistory()从tushare上获取数据并保存在MySql数据库中，
    mysql2RawData()从MySql提取数据并转换为RawData类。
B、charVal.py：
    将原始数据转换为10大类型共58个特征值（量化指标），作为后续分析数据。
C、dataset.py：
    特征数据转换为Dataset类，利用next_batch迭代随机输入batch数据进行训练与预测。
D、LSTMgraph.py：
    LSTM 模型的 tensorflow 流程图。
E、train_predict.py：
    模型训练与预测。
F、Results：
    保存模型训练的结果，保存在以"stock-[code|股票代码]"命名的文件夹里。
    每个文件夹对应一个股票。


