import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10 as CIFARDataset
from bai7_cifar10_difine_aNN import SimpleNeuralNetwork
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.optim import Adam
from sklearn.metrics import classification_report 

if __name__ == "__main__":
    train_dataset = CIFARDataset(root="./data", train=True, download=True, transform=ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    test_dataset = CIFARDataset(root="./data", train=False, download=True, transform=ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = SimpleNeuralNetwork(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    
    epochs = 10
    for epoch in range(epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            indices = torch.argmax(predictions, dim=1)
            all_predictions.extend(indices.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    print(f"\nEvaluation after {epochs} epochs:")
    print(classification_report(all_labels, all_predictions, target_names=test_dataset.classes))

