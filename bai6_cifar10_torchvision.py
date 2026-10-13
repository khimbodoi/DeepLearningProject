from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
if __name__ == "__main__":
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=ToTensor())
    img, label = train_dataset.__getitem__(34)
    print(label)
    print(img.shape)
    
    training_dataloaders = DataLoader(
        dataset=train_dataset, 
        batch_size=16,
        num_workers=4,
        shuffle=True,
        drop_last=True
    )
        
    for img, label in training_dataloaders:
        print(img.shape)
        print(label)
         