from torch.utils.data import Dataset
import os
from PIL import Image
import urllib.request
import tarfile
import pickle
import numpy as np
import cv2
from torchvision.transforms import ToTensor, Resize, Compose
from torch.utils.data import DataLoader


class AnimalDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.transform = transform
        if train: 
            mode = "train"
        else:
            mode = "test"

        self.root = os.path.join(root, mode)
        self.categories = ["cat", "dog", "cow", "horse", "chicken", "squirrel", "sheep", "elephant", "spider", "butterfly"]

        self.images = []
        self.labels = []
        for i, category in enumerate(self.categories):
            data_files_path = os.path.join(self.root, category)
            if os.path.exists(data_files_path):
                for img_name in os.listdir(data_files_path):
                    img_path = os.path.join(data_files_path, img_name)
                    self.images.append(img_path)
                    self.labels.append(i)
            else:
                print(f"Warning: Category path {data_files_path} does not exist.")
    
    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

if __name__ == "__main__":
    root = r"D:\Download\data"

    transform = Compose([Resize((200, 200)), ToTensor()])
    dataset = AnimalDataset(root, train=True, transform=transform)
    
    training_dataloaders = DataLoader(
        dataset=dataset, 
        batch_size=64,
        num_workers=4,
        shuffle=True,
        drop_last=True
    )

    for img, label in training_dataloaders:
        print(img.shape)
        print(label)
