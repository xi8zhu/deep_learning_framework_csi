# Problem Document

1. motivation mse极小而模型参数很大，质疑是否需要那么大的参数来预测这样一个问题
+ 选择更加轻量的空时频预测模型
+ 只预测单载波单接收天线并解决泛化性
+ 这个motivation不太成立，mse应该改成l2_norm，单个模型参数对模型的贡献不大，所以可能可以考虑简化模型

2. 发现test性能比train好 --不合理 --怀疑是sum_loss在训练过程溢出，待调试
+ 修改model.train()的位置
+ 查阅发现差距不大是正常现象，dropout带来的影响或数据集不平衡
  + 数据集采用random split划分，可以判定应该是不存在不平衡的问题
  + 重新做了组7比3的实验，仍然test性能较train好
    + 就算SGCS平均有问题，但mse应该是不存在这个问题的。
+ 最终倾向于dropout的问题--或许要看看model的设计
  + 模型未使用dropout层，也未使用batchnorm
+ 只能说数据量小计算的指标性能好....
+ GPT说可能是模型欠拟合，感觉有道理，"训练集中的数据随机性较大"构建也有可能
+ GPT还说增强轮次，那目前来说训练轮次增多会让训练集和测试集的性能无比接近，这是好事！
https://www.zhihu.com/question/429337764#:~:text=model.trai
https://www.zhihu.com/question/264677004#:~:text=%E6%AD%A4%E6%97%B6%EF%BC%8C%E6%9C%89%E5%8F%AF%E8%83%BD%E9%AA%8C%E8%AF%81%E9%9B%86%E6%88%96