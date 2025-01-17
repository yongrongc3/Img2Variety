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
class MobileNetV2(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV2, self).__init__()
        self.mobilenet_v2 = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        in_features = self.mobilenet_v2.classifier[1].in_features
        self.mobilenet_v2.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.mobilenet_v2(x)
        return x

model = MobileNetV2(num_classes=93).to(device)
model.load_state_dict(torch.load("/models/baseline+pretain_MobilenetV2_XXX_{epoch}.pth"))
criterion = nn.CrossEntropyLoss()

# test model
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
print("Test Loss: {:.4f}, Accuracy: {:.4f}%".format(test_loss, test_acc * 100))
# precision/recall/f1-score
precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
recall = recall_score(y_true, y_pred,average='macro')
f1 = f1_score(y_true, y_pred,average='macro')
print(f'Precision: {precision}, Recall: {recall}, F1 Score: {f1}')

# Define the file path
csv_file = './results/test_baseline+pretain_MobilenetV2_XXX_pred.csv'
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image Name', 'Image Label', 'Predicted Label'])
    for name, true_label, pred_label in zip(image_names, y_true, y_pred):
        writer.writerow([name, true_label, pred_label])