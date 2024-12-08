import matplotlib.pyplot as plt
import numpy as np
if __name__ == '__main__':

    # 数据准备
    x = [1,2,3,4,5]  # 在 0 到 10 之间生成 100 个点
    y = [1,2,3,4,5]        # y 为 x 的正弦值

    # 创建图形
    plt.figure()

    # 绘制折线图
    plt.scatter(x, y, label='sin(x)', linewidth=2, c = 'r')
    y = [1,5,3,8,5]        # y 为 x 的正弦值
    plt.scatter(x, y, label='sin(x)', linewidth=2)
    y = [1,2,3,7,8]        # y 为 x 的正弦值
    plt.scatter(x, y, label='sin(x)', linewidth=2)
    y = [5,2,3,4,5]        # y 为 x 的正弦值
    plt.scatter(x, y, label='sin(x)', linewidth=2)

    # # 添加标题和标签
    plt.title("Sine Wave", fontsize=16)
    # plt.xlabel("x-axis", fontsize=14)
    # plt.ylabel("y-axis", fontsize=14)

    # # 添加网格和图例
    # plt.grid(True, linestyle='--', alpha=0.7)
    # plt.legend(fontsize=12)

    # # 显示图形
    plt.show()
