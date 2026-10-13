import torch 
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# ResNet Bottleneck Block
class Bottleneck(nn.Module):
    # Trong ResNet, số kênh đầu ra luôn gấp 4 lần số kênh ở giữa khối Bottleneck
    expansion = 4 

    def __init__(self, in_channels, mid_channels, stride=1, downsample=None):
        super().__init__()
        
        out_channels = mid_channels * self.expansion
        
        # 1. Conv 1x1: Giảm chiều kênh
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU()
        )
        
        # 2. Conv 3x3: Trích xuất đặc trưng. Lưu ý tham số stride!
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False),            
            nn.BatchNorm2d(mid_channels),
            nn.ReLU()
        )

        # 3. Conv 1x1: Tăng số chiều kênh
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
       
        self.relu_final = nn.ReLU()
        
        self.downsample = downsample 

    def forward(self, x):
        identity = x
        # x = [256 kênh, 56x56]
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        # XỬ LÝ NHÁNH TẮT: Nếu có cấu trúc downsample truyền vào, x sẽ chạy qua nó trước khi cộng
        ## out = [256 kênh, 56x56] đạt
        ## out = [256 kênh, 28x28] không đạt = x -> qua lớp 1x1 tride=2 để ép kiểu về [256 kênh, 56x56]
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu_final(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks_list, num_classes=1000):
        super().__init__()
        # Số kênh đầu ra chuẩn sau khi đi qua phần Stem (chưa tạo) luôn là 64
        self.in_channels = 64

        # (Phần Stem - Khai báo ở Bước 3)
        
        # Tạo 4 Stage bằng cách gọi hàm _make_layer
        # Ví dụ ResNet-50: num_blocks_list = [3, 4, 6, 3]
        self.layer1 = self._make_layer(block, mid_channels=64,  num_blocks=num_blocks_list[0], stride=1)
        self.layer2 = self._make_layer(block, mid_channels=128, num_blocks=num_blocks_list[1], stride=2)
        self.layer3 = self._make_layer(block, mid_channels=256, num_blocks=num_blocks_list[2], stride=2)
        self.layer4 = self._make_layer(block, mid_channels=512, num_blocks=num_blocks_list[3], stride=2)



    if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BottleneckBlock().to(device)

    dummy_input = torch.randn(2, 256, 56, 56).to(device) 
    output = model(dummy_input)
    
    print("Model architecture:\n", model)
    print("Output tensor shape:", output.shape)
