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

