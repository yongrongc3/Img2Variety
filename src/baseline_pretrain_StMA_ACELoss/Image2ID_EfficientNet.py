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

class EfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNet, self).__init__()
        self.efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.efficientnet(x)
        return x

# Modify the classifier to fit the number of classes 
# 93 rice accessions identification
model = EfficientNet(num_classes=93).to(device)
# 224 maize lines identification
# model = EfficientNet(num_classes=224).to(device)
# 2 rice subspecies classification
# model = EfficientNet(num_classes=2).to(device)

# Count the total number of parameters in the model
total_params = sum(p.numel() for p in model.parameters())
print(">>> Total params: {:.2f} M".format(total_params / 1e6))

class WeightedLoss(nn.Module):
    def __init__(self, alpha=1.2):
        super(WeightedLoss, self).__init__()
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        correct = torch.argmax(inputs, dim=1) == targets
        weighted_loss = ce_loss * (1 + self.alpha * (~correct).float())
        return weighted_loss.mean()
criterion = WeightedLoss(alpha=0.5)
# criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

def Draw_loss(losses):
    plt.title("Training Loss")
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig("./results/loss_Image2ID_EfficientNet_XXX.pdf")

losses = []

for epoch in range(150):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        if isinstance(outputs, tuple):  
            outputs = outputs[0]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    losses.append(running_loss / len(train_dataset))
    print('Epoch: [%d] loss: %.6f' % (epoch + 1, running_loss * 64 / len(train_dataset)))
    if epoch > 140:
        save_folder = './models'
        save_path = os.path.join(save_folder, f'loss_Image2ID_EfficientNet_XXX_{epoch}.pth')
        torch.save(model.state_dict(), save_path)
        model.eval()
        test_loss = 0.0
        test_acc = 0.0
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
                y_score.extend(probs.cpu().numpy())

        # accuracy
        test_loss /= len(test_loader.dataset)
        test_acc = accuracy_score(y_true, y_pred)
        # print("Test Loss: {:.4f}, Accuracy: {:.4f}%".format(test_loss, test_acc * 100))
        # precision/recall/f1-score
        precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
        recall = recall_score(y_true, y_pred,average='macro')
        f1 = f1_score(y_true, y_pred,average='macro')
        # print(f'Precision: {precision}, Recall: {recall}, F1 Score: {f1}')
        print(f"Epoch:{epoch}\tTest Loss: {test_loss:.4f}\tAccuracy: {test_acc * 100:.4f}%\tPrecision: {precision* 100:.4f}%\t, Recall: {recall* 100:.4f}%\t, F1 Score: {f1* 100:.4f}%\t")
        csv_filename = './results/loss_Image2ID_EfficientNet_XXX.csv'
        header = ['Epoch', 'Test Loss', 'Accuracy (%)', 'Precision(%)', 'Recall(%)', 'F1 Score(%)']
        row = [f"{epoch}, "f"{test_loss:.4f}", f"{test_acc * 100:.4f}", f"{precision* 100:.4f}", f"{recall* 100:.4f}", f"{f1* 100:.4f}"]

        file_exists = os.path.isfile(csv_filename)

        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            if not file_exists:
                writer.writerow(header)
            
            writer.writerow(row)
Draw_loss(losses)
