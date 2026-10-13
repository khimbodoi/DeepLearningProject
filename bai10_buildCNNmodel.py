import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from bai5_animal_dataset import AnimalDataset
from torchvision.transforms import Compose, Resize, ToTensor
from sklearn.metrics import classification_report, accuracy_score
from argparse import ArgumentParser
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import shutil

def get_arguments():
    parser = ArgumentParser(description="Training Simple CNN")
    parser.add_argument("--num_epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--root", type=str, default="./animal", help="Root directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes")
    parser.add_argument("--log_dir", type=str, default="./tensorboard_log", help="Tensorboard log directory")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--checkpoint_file", type=str, default="train_animal_model.pth", help="Checkpoint file")
    args = parser.parse_args()
    return args

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = self.make_block(in_channels=3, out_channels=8)
        self.conv2 = self.make_block(in_channels=8, out_channels=16)
        self.conv3 = self.make_block(in_channels=16, out_channels=32)
        self.conv4 = self.make_block(in_channels=32, out_channels=64)
        self.conv5 = self.make_block(in_channels=64, out_channels=128)
        
        self.fc1 = nn.Sequential (
            nn.Dropout(p=0.5),
            nn.Linear(128 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def make_block(self, in_channels, out_channels):
        
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = nn.Flatten()(x)
        x = self.fc1(x)
        return x


if __name__ == "__main__":
    args = get_arguments()
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Định nghĩa Transform (bắt buộc Resize về 224x224 để khớp với FC layer)
    transform = Compose([
        Resize((224, 224)),
        ToTensor()
    ])

    # 3. Nạp Dataset và DataLoader cho cả Train và Test
    root = "./animal"  # hoặc r"D:\Download\data"
    train_dataset = AnimalDataset(root=root, train=True, transform=transform)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        drop_last=True
    )

    test_dataset = AnimalDataset(root=root, train=False, transform=transform)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        drop_last=False
    )

    if os.path.isdir(args.checkpoint_dir):
        shutil.rmtree(args.checkpoint_dir)
    if not os.path.isdir(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)

    # 4. Khởi tạo mô hình, hàm Loss và Optimizer
    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_iters = len(train_dataloader)
    #tensorboard
    writer = SummaryWriter(args.log_dir)

    best_acc = 0.0
    all_predictions = []
    all_labels = []
    # 5. Vòng lặp huấn luyện (Training Loop)
    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, colour="#40E0D0")
        for i, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss_value = criterion(outputs, labels)

            progress_bar.set_description(f"Epoch {epoch+1}/{num_epochs}. Iteration {i+1}/{num_iters}. Loss {loss_value.item():.3f}")
            # TensorBoard ghi log Loss
            writer.add_scalar("Loss/train", loss_value.item(), epoch * num_iters + i)
            
            # Backward pass & Optimize
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

        # 6. Đánh giá mô hình sau mỗi Epoch (Evaluation)
        model.eval()
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for images, labels in test_dataloader:
                images, labels = images.to(device), labels.to(device)
                predictions = model(images)
                indices = torch.argmax(predictions, dim=1)
                all_predictions.extend(indices.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        accuracy = accuracy_score(all_labels, all_predictions)
        print("Epoch {}: Accuracy: {}".format(epoch + 1, accuracy))
        writer.add_scalar("Accuracy/test", accuracy, epoch + 1)

        # Lưu checkpoint cuối cùng
        last_path = os.path.join(args.checkpoint_dir, "last_CNN.pt")
        torch.save(model.state_dict(), last_path)

        # Lưu checkpoint tốt nhất
        if accuracy > best_acc:
            best_acc = accuracy
            best_path = os.path.join(args.checkpoint_dir, "best_CNN.pt")
            torch.save(model.state_dict(), best_path)
            print("Đã lưu mô hình tốt nhất (Accuracy = {:.4f}) tại: {}".format(best_acc, best_path))

    print("\n--- Báo cáo đánh giá (Classification Report) ---")
    print(classification_report(all_labels, all_predictions, target_names=train_dataset.categories))
    writer.close()
    print("Finished Training")