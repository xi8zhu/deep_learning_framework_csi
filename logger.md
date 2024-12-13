# Logger
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