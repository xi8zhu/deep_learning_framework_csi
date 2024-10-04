# 自用AI框架
## Todo
+ 完善recorder
+ 添加模型的测试文件
+ 修改代码，使得代码参数可在配置文件中调整
+ 添加不同范式的训练（只取一只接收天线和一只子载波）
+ 对不同的训练提供不同的测试
+ 添加学习率衰减

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
+ runs:模型参数，测试用到，训练模型未使用（用log替代了）
