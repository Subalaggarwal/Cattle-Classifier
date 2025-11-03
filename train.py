import os
import shutil
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import CattleDataset
from model import create_resnet50


root_dir = r'C:\\Users\\ACER\\Desktop\\Cattle Breed Classifier\\data'
split_root = r'C:\\Users\\ACER\\Desktop\\Cattle Breed Classifier\\final_data'
train_dir = os.path.join(split_root, 'train')
val_dir = os.path.join(split_root, 'val')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")


classes = sorted(os.listdir(root_dir))
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

image_paths, labels = [], []
for cls in classes:
    class_folder = os.path.join(root_dir, cls)
    if not os.path.isdir(class_folder): continue
    imgs = [os.path.join(class_folder, f) for f in os.listdir(class_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_paths.extend(imgs)
    labels.extend([class_to_idx[cls]] * len(imgs))

train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(
    image_paths, labels, test_size=0.2, stratify=labels, random_state=42
)

train_ds = CattleDataset(train_imgs, train_lbls, train=True)
val_ds = CattleDataset(val_imgs, val_lbls, train=False)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

model = create_resnet50(num_classes=len(classes), pretrained=True).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
best_val_acc = 0.0
train_losses, val_losses, val_accuracies = [], [], []

for epoch in range(15):
    model.train()
    running_loss, correct, total = 0, 0, 0
    for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == lbls).sum().item()
        total += lbls.size(0)

    train_acc = correct / total
    avg_train_loss = running_loss / total
    train_losses.append(avg_train_loss)


    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            val_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            val_correct += (preds == lbls).sum().item()
            val_total += lbls.size(0)
            y_true.extend(lbls.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    val_acc = val_correct / val_total
    avg_val_loss = val_loss / val_total
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}/{15} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

    #
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({'model_state': model.state_dict(), 'classes': classes}, 'best_model.pth')
        print(f"New best model saved with val_acc = {val_acc:.4f}")

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=classes))

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(val_accuracies, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Validation Accuracy')
plt.show()
