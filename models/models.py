import torch.nn as nn
import torchvision.models as models

# ResNet-18モデルの定義
class CustomResNet18(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet18, self).__init__()
        self.resnet = models.resnet18(weights=None) #"IMAGENET1K_V1")  # 事前学習済みのResNet-18モデルをロード
        # 最終の全結合層を入れ替えて、出力クラスの数に合わせる
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


# ResNet-18モデルの定義
class CustomResNet32(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet32, self).__init__()
        self.resnet = models.resnet32(weights=None) #"IMAGENET1K_V1")  # 事前学習済みのResNet-18モデルをロード
        # 最終の全結合層を入れ替えて、出力クラスの数に合わせる
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
    
class CustomEfficientNetb0(nn.Module):
    def __init__(self, num_classes):
        super(CustomEfficientNetb0, self).__init__()
        self.efficientnet = models.efficientnet_b0(weights=None) 
        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(in_features, num_classes)  # 新しい最終層を追加

    def forward(self, x):
        return self.efficientnet(x)

class CustomEfficientNetb1(nn.Module):
    def __init__(self, num_classes):
        super(CustomEfficientNetb1, self).__init__()
        self.efficientnet = models.efficientnet_b1(weights=None) 
        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(in_features, num_classes)  # 新しい最終層を追加

    def forward(self, x):
        return self.efficientnet(x)

class CustomEfficientNetb2(nn.Module):
    def __init__(self, num_classes):
        super(CustomEfficientNetb2, self).__init__()
        self.efficientnet = models.efficientnet_b2(weights=None) 
        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(in_features, num_classes)  # 新しい最終層を追加

    def forward(self, x):
        return self.efficientnet(x)
