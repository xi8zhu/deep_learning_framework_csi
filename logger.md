# Logger
### embedding没法输入float类型
+ 换成linear

### mask的设置
+ encoding的mask设成ones(batch_size, )
+ decoding的mask直接调用subsequent_mask()函数