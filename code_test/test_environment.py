import torch
from tensorboardX import SummaryWriter



if __name__ == '__main__':
    print("hello world")
    print("PyTorch 版本:", torch.__version__)
    
    # 检查是否可以使用 GPU
    if torch.cuda.is_available():
        print("CUDA 可用！")
        print("GPU 设备:", torch.cuda.get_device_name(0))
    else:
        print("CUDA 不可用，使用 CPU。")
    
    # 检查 PyTorch 张量是否能在 GPU 上运行
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.randn(3, 3, device=device)
    print("张量示例：", tensor)