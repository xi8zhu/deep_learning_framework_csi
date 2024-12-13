# 自用AI框架
## Todo
+ ConvLSTM可以使用config调节网络参数
+ sgcs写得更优雅一点,对于不同维度的输入
+ 或许可以尝试实施的记录训练结果在另一个txt上
+ 添加不同范式的训练（只取一只接收天线和一只子载波）
+ 添加学习率衰减
+ 添加多线程
### 数据规范
在CSI任务中, 一般出现的数据维度为

`
(Batch_size, time_slots, subcarrier, tx, rx, real_imag)
`
#### CSI预测
| data_settings      | tensor_dimension |
| :-----------: | :-----------: |
| 0      | `(Batch_size, time_slots, tx)`        |
| 1   | `(Batch_size, time_slots, channels = rx * real_imag, sb, tx)`        | 

#### CSI反馈 
+ 通常使用一根接收天线或特征向量反馈时, 数据维度为`(Batch_size, time_slots, subcarrier, tx, real_imag)`
+ 若不考虑时间相关性, 数据维度为`(Batch_size,  subcarrier, tx, rx, real_imag)`
  + 若二者均不考虑,则`(Batch_size,  subcarrier, tx, real_imag)`
  + 对于transformer模型, 会将维度flatten,得`(Batch_size,  subcarrier * tx * real_imag)`

  + 对于CNN相关的模型, 把subcarrier与tx看作图像的高和宽, 将数据重构为:`(Batch_size, real_imag * rx, subcarrier, tx)`
### 程序运行说明
+ program中为程序的大部分细节文件
+ 创建环境
```
conda env create -f environment.yaml
```
+ 可在.yaml文件中更改所有相关配置
```
python main.py --config config/csi_prediction/train.yaml
```

+ 使用tensorboard查看模型损失
```
tensorboard --logdir log/csi_prediction
```
### 输出结果说明
+ log：tensorboard保存结果
+ checkpoints: 模型参数文件
+ results:输出模型的配置，模型的各种测试结果并打印了一份模型
