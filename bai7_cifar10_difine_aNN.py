import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
class SimpleNeuralNetwork(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(32 * 32 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes) 
        )


    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        return x

if __name__ == "__main__":
    model = SimpleNeuralNetwork()
    import_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms.ToTensor())
    data_tensor = torch.tensor(import_data.data, dtype=torch.float32)
    results = model(data_tensor)
    print(results.shape) # Should print torch.Size([50000, 10]) since CIFAR-10 has 50,000 training images and 10 classes

