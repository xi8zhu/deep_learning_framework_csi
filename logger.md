# Logger

## 2024.12.20
+ 调参模型效果越来越差。
  + 调高LSTM的层数（6），损失曲线波动极大，无法正确学到特征
  + 学习率等特征改变，发现罪魁祸首在l2正则化，当轮数变大时，l2正则化损失稳定下降而其他两个损失波动极大，失去学习能力
    + 正则化的损失权重看来要最后调整。---第一波训练似乎证明正则化有用
+ 最近总是出现GPU过度断联的情况。---两次
  + `Unable to determine the device handle for GPU 0000:02:00.0: Unknown Error`
  + sudo reboot重启可解决
  + 杜绝此问题或许可尝试:https://blog.csdn.net/yu_xiao_you/article/details/130948104
  + 尚未尝试

## 2024.12.14
+ 跑完一遍TransLSTM，效果提升不大
  + 计划：尝试多层LSTM
  + 计划：设计csi预测的数据读入范式

## 2024.12.13
+ 完成TransLSTM的设计与代码实现，成功跑通。
  + 解决embedding的换成linear
  + 完成transformer与LSTM的级联
  + 更改了读入数据集的方式
  + 更改了计算sgcs的方式（输入四个数和两个数的区别）
+ config的设置改变，ConvLSTM的参数也可以被输入（但现在尚无法被改变）

## 2024.12.9
### 探究transformer层的embedding实现
+ 发现embedding没法输入float类型
+ 后续需要对embedding进行改进

### mask的设置
+ encoding的mask设成ones(batch_size, )
+ decoding的mask直接调用subsequent_mask()函数
