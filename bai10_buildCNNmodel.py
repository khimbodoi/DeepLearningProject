import os
import shutil
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from bai5_animal_dataset import AnimalDataset
from torchvision.transforms import Compose, Resize, ToTensor, RandomHorizontalFlip
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from argparse import ArgumentParser
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def get_args():
    parser = ArgumentParser(description="CNN training")
    parser.add_argument("--root", "-r", type=str, default="./animal", help="Root of the dataset")
    parser.add_argument("--epochs", "--num_epochs", "-e", type=int, default=100, dest="epochs", help="Number of epochs")
    parser.add_argument("--batch-size", "--batch_size", "-b", type=int, default=64, dest="batch_size", help="Batch size")
    parser.add_argument("--image-size", "--image_size", "-i", type=int, default=224, dest="image_size", help="Image size")
    parser.add_argument("--logging", "--log_dir", "-l", type=str, default="tensorboard", dest="logging")
    parser.add_argument("--trained_models", "-t", type=str, default="trained_models")
    parser.add_argument("--checkpoint", "-c", type=str, default=None)
    args = parser.parse_args()
    return args

def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="Wistia")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    threshold = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

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


if __name__ == '__main__':
    args = get_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    train_transform = Compose([
        Resize((args.image_size, args.image_size)),
        RandomHorizontalFlip(p=0.5),
        ToTensor(),
    ])
    test_transform = Compose([
        Resize((args.image_size, args.image_size)),
        ToTensor(),
    ])
    train_dataset = AnimalDataset(root=args.root, train=True, transform=train_transform)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    test_dataset = AnimalDataset(root=args.root, train=False, transform=test_transform)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )
    if os.path.isdir(args.logging):
        shutil.rmtree(args.logging)
    if not os.path.isdir(args.trained_models):
        os.mkdir(args.trained_models)
    writer = SummaryWriter(args.logging)
    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint.get("best_acc", 0)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        start_epoch = 0
        best_acc = 0

    num_iters = len(train_dataloader)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, colour="green")
        for iter, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            # forward
            outputs = model(images)
            loss_value = criterion(outputs, labels)
            progress_bar.set_description("Epoch {}/{}. Iteration {}/{}. Loss {:.3f}".format(epoch+1, args.epochs, iter+1, num_iters, loss_value))
            writer.add_scalar("Train/Loss", loss_value, epoch*num_iters+iter)
            # backward
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

        model.eval()
        all_predictions = []
        all_labels = []
        for iter, (images, labels) in enumerate(test_dataloader):
            all_labels.extend(labels)
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                predictions = model(images)   # predictions shape 64x10
                indices = torch.argmax(predictions.cpu(), dim=1)
                all_predictions.extend(indices)
                loss_value = criterion(predictions, labels)
        all_labels = [label.item() for label in all_labels]
        all_predictions = [prediction.item() for prediction in all_predictions]

        cm = confusion_matrix(all_labels, all_predictions)
        figure = plot_confusion_matrix(cm, class_names=test_dataset.categories)
        writer.add_figure('confusion_matrix', figure, epoch)

        accuracy = accuracy_score(all_labels, all_predictions)
        print("Epoch {}: Accuracy: {}".format(epoch+1, accuracy))
        writer.add_scalar("Val/Accuracy", accuracy, epoch)
        # torch.save(model.state_dict(), "{}/last_cnn.pt".format(args.trained_models))
        checkpoint = {
            "epoch": epoch+1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(checkpoint, "{}/last_cnn.pt".format(args.trained_models))
        if accuracy > best_acc:
            checkpoint = {
                "epoch": epoch + 1,
                "best_acc": accuracy,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(checkpoint, "{}/best_cnn.pt".format(args.trained_models))
            best_acc = accuracy
        # print(classification_report(all_labels, all_predictions))

    print("\n--- Báo cáo đánh giá (Classification Report) ---")
    print(classification_report(all_labels, all_predictions, target_names=train_dataset.categories))
    writer.close()
    print("Finished Training")