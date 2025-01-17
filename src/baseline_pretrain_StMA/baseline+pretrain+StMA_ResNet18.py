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
# Count the total number of parameters in the model
total_params = sum(p.numel() for p in model.parameters())
# print(f"Number of parameters: {total_params}")
print(">>> Total params: {:.2f} M".format(sum(p.numel() for p in model.parameters()) / 1e6))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
def Draw_loss(losses):
    plt.title("Training Loss")
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig("./results/loss_baseline+pretrain+StMA_ResNet18_XXX.pdf")
losses = []

for epoch in range(150):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    losses.append(running_loss / len(train_dataset))
    print('Epoch: [%d] loss: %.6f' % (epoch + 1, running_loss * 64 / len(train_dataset)))
    if epoch > 148:
        save_folder = '/models'
        save_path = os.path.join(save_folder, f'baseline+pretrain+StMA_ResNet18_XXX_{epoch}.pth')
        torch.save(model.state_dict(), save_path)
Draw_loss(losses)

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
print("Test Loss: {:.4f}, Accuracy: {:.4f}%".format(test_loss, test_acc * 100))
# precision/recall/f1-score
precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
recall = recall_score(y_true, y_pred,average='macro')
f1 = f1_score(y_true, y_pred,average='macro')
print(f'Precision: {precision}, Recall: {recall}, F1 Score: {f1}')

# Define the file path
csv_file = './results/baseline+pretrain+StMA_ResNet18_XXX_pred.csv'
# Write y_true and y_pred to CSV
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['y_true', 'y_pred'])  # Write header
    for true, pred in zip(y_true, y_pred):
        writer.writerow([true, pred])

cm = confusion_matrix(y_true, y_pred)
plt.imshow(cm, cmap=plt.cm.Blues)
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
class_labels = test_dataset.classes
tick_marks = range(len(class_labels))  
plt.xticks(tick_marks, class_labels, rotation=45, fontsize=2)
plt.yticks(tick_marks, class_labels, fontsize=2)
plt.savefig('./results/baseline+pretrain+StMA_ResNet18_XXX_ConfusionMatrix.pdf', format='pdf')


