import os
import torch
import random
import csv
import numpy as np
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score
from matplotlib.font_manager import FontProperties
import torchvision.models as models
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, average_precision_score, roc_curve, confusion_matrix, auc
from sklearn.preprocessing import label_binarize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_random_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value) 
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  

set_random_seed()
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
train_dataset = ImageFolder('./data/xxxx/train', transform=train_transforms)
test_dataset = ImageFolder('./data/xxxx/test', transform=test_transforms)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4) 
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4) 
class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Linear(512, num_classes)
    def forward(self, x):
        x = self.resnet(x)
        return x
# Modify the classifier to fit the number of classes 
# 93 rice accessions identification
model = ResNet(num_classes=93).to(device)
# 224 maize lines identification
# model = ResNet(num_classes=224).to(device)
# 2 rice subspecies classification
# model = ResNet(num_classes=2).to(device)
class WeightedLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super(WeightedLoss, self).__init__()
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        correct = torch.argmax(inputs, dim=1) == targets
        weighted_loss = ce_loss * (1 + self.alpha * (~correct).float())
        return weighted_loss.mean()
criterion = WeightedLoss(alpha=1)
for epoch in range(141, 150):
    model.load_state_dict(torch.load(f'/models/baseline_pretrain_ACELoss_ResNet_XXX_{epoch}.pth'))
    # criterion = nn.CrossEntropyLoss()

    # 测试模型
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    image_names = []
    y_true = []
    y_pred = []
    y_score = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())
            image_names.extend(test_dataset.imgs[i][0].split('/')[-1] for i in range(len(test_dataset.imgs)))
            y_score.extend(probs.cpu().numpy())

    # accuracy
    test_loss /= len(test_loader.dataset)
    test_acc = accuracy_score(y_true, y_pred)
    print(f"epcoh:{epoch}")
    print("Test Loss: {:.4f}, Accuracy: {:.4f}%".format(test_loss, test_acc * 100))
    # precision/recall/f1-score
    precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_true, y_pred,average='macro')
    f1 = f1_score(y_true, y_pred,average='macro')
    print(f'Precision: {precision}, Recall: {recall}, F1 Score: {f1}')

# Define the file path
csv_file = './results/test_baseline_pretrain_ACELoss_ResNet_XXX_pred.csv'
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image Name', 'Image Label', 'Predicted Label'])
    for name, true_label, pred_label in zip(image_names, y_true, y_pred):
        writer.writerow([name, true_label, pred_label]) 