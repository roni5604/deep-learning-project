import os
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score

import torchvision.transforms as T
from torch.optim.lr_scheduler import StepLR

class ExpressionDatasetCNN(Dataset):
    """
    Loads images as single-channel, applies data augmentation for training, basic transform for validation.
    """
    def __init__(self, base_dir, is_train=True, img_size=(48,48)):
        self.samples = []
        self.labels = []
        self.class_to_idx = {}
        self.img_size = img_size
        self.is_train = is_train

        if not os.path.exists(base_dir):
            print(f"[ERROR] Directory not found: {base_dir}")
            return

        subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        subdirs.sort()
        for idx, c in enumerate(subdirs):
            self.class_to_idx[c] = idx
            class_dir = os.path.join(base_dir, c)
            files = glob.glob(os.path.join(class_dir, "*"))
            for f in files:
                self.samples.append((f, idx))

        # Data augmentation for training
        if is_train:
            self.transform = T.Compose([
                T.Grayscale(num_output_channels=1),  # ensure 1-channel
                T.Resize(img_size),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=10),
                # Potentially T.RandomCrop(48, padding=4) for more variety
                T.ToTensor()
            ])
        else:
            self.transform = T.Compose([
                T.Grayscale(num_output_channels=1),
                T.Resize(img_size),
                T.ToTensor()
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        with Image.open(img_path) as img:
            # Convert to grayscale + augment
            x = self.transform(img)
            # shape => (1, H, W)
        return x, torch.tensor(label)

class AdvancedCNN(nn.Module):
    """
    A deeper CNN with multiple conv blocks, dropout, and linear layers.
    """
    def __init__(self, num_classes=7):
        super(AdvancedCNN, self).__init__()
        # Block 1: (1 -> 32)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(p=0.25)

        # Block 2: (32 -> 64)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(p=0.25)

        # Block 3: (64 -> 128)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout2d(p=0.25)

        # After 3 pools of stride=2 on 48x48 => 48->24->12->6
        # so final feature map is (128, 6, 6) => 128*6*6 = 4608
        self.fc1 = nn.Linear(128*6*6, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Block 1
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.drop2(x)

        # Block 3
        x = F.relu(self.conv5(x))
        x = self.pool3(x)
        x = self.drop3(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # FC
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec  = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    return acc, prec, rec

def main():
    print("\n=== Advanced CNN Model with Data Augmentation ===")

    train_dir = "data/face-expression-recognition-dataset/images/images/new_train"
    val_dir   = "data/face-expression-recognition-dataset/images/images/new_validation"
    batch_size = 64
    num_epochs = 15
    learning_rate = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Create datasets
    train_dataset = ExpressionDatasetCNN(train_dir, is_train=True,  img_size=(48,48))
    val_dataset   = ExpressionDatasetCNN(val_dir,   is_train=False, img_size=(48,48))

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("[ERROR] Dataset is empty. Exiting advanced CNN.")
        return

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    num_classes = len(train_dataset.class_to_idx)
    model = AdvancedCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Simple step LR scheduler
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    print("Starting CNN training (with random flips, rotations, dropout, etc.)...\n")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Step the scheduler
        scheduler.step()

        # Evaluate
        val_acc, val_prec, val_rec = evaluate_model(model, val_loader, device)
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train_Loss: {running_loss/len(train_loader):.4f} | "
              f"Val_Acc: {val_acc:.4f} | "
              f"Val_Prec: {val_prec:.4f} | "
              f"Val_Rec: {val_rec:.4f}")

    # Final
    val_acc, val_prec, val_rec = evaluate_model(model, val_loader, device)
    print("\n=== Final Evaluation on Validation Set ===")
    print(f"Accuracy:  {val_acc:.4f}")
    print(f"Precision: {val_prec:.4f}")
    print(f"Recall:    {val_rec:.4f}")

    with open("cnn_results.txt", "w") as f:
        f.write(f"{val_acc:.4f},{val_prec:.4f},{val_rec:.4f}")

    print("\nAdvanced CNN training complete! Results saved to 'cnn_results.txt'.")

if __name__ == "__main__":
    main()
