from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split


def Create_dataset(img_size, dir):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)), # 画像サイズをリサイズ,
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # データセットを作成
    dataset = datasets.ImageFolder(dir, transform=transform)
    return dataset


def Create_Dataloader(img_size, dir, batch_size):
    dataset = Create_dataset(img_size, dir)
    # 学習データに使用する割合
    n_train_ratio = 0.9

    # 割合から個数を出す
    n_train = int(len(dataset) * n_train_ratio)
    n_val   = int(len(dataset) - n_train)

    # 学習データと検証データに分割
    train, val = random_split(dataset, [n_train, n_val])

    # Data Loader
    train_loader = DataLoader(train, batch_size, shuffle=True, drop_last=True, num_workers=4)
    val_loader = DataLoader(val, batch_size, num_workers=4)
    return train_loader, val_loader
