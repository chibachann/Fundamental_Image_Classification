import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataloader.dataloader import Create_Dataloader
from models.resnet18 import CustomResNet18
from train import train_and_evaluate_model

# 画像のサイズとデータのディレクトリのパス
IMG_SIZE = 224
FLOWER_DIR = 'C:/Users/tairy/Desktop/work/Flower Recognition/input/flowers'

# バッチサイズとエポック数
batch_size = 128
num_epoch = 10

def main(device, img_size, flower_dir, batch_size, num_epoch):
    train_loader, val_loader = Create_Dataloader(img_size, flower_dir, batch_size)

    model = CustomResNet18(num_classes=5).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)

    train_losses, val_losses, train_accuracies, val_accuracies = train_and_evaluate_model(model, train_loader, val_loader, criterion, optimizer, scheduler,num_epoch)

    # LossとAccuracyのグラフを表示
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    main(device, IMG_SIZE, FLOWER_DIR, batch_size, num_epoch)
