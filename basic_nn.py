import os
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

# Optional: If you want a scheduler
from torch.optim.lr_scheduler import StepLR


class ExpressionDatasetMLP(Dataset):
    """
    Loads images, flattens them, and holds numeric labels for classification.
    """

    def __init__(self, base_dir, img_size=(48, 48), scaler=None):
        self.samples = []
        self.labels = []
        self.class_to_idx = {}
        self.img_size = img_size
        self.scaler = scaler

        if not os.path.exists(base_dir):
            print(f"[ERROR] Directory not found: {base_dir}")
            return

        subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        subdirs.sort()
        for idx, c in enumerate(subdirs):
            self.class_to_idx[c] = idx
            class_dir = os.path.join(base_dir, c)
            image_files = glob.glob(os.path.join(class_dir, "*"))
            for f in image_files:
                self.samples.append((f, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        with Image.open(img_path).convert('L') as img:
            img = img.resize(self.img_size)
            arr = np.array(img, dtype=np.float32).reshape(-1)

        if self.scaler is not None:
            # We'll manually apply the scaler transform
            arr = self.scaler.transform([arr])[0]

        # Normalize from [0,255] to [0,1] was done if the scaler is none
        # but if the scaler is present, we assume the user already did it
        return torch.tensor(arr), torch.tensor(label)


class DeeperMLP(nn.Module):
    """
    MLP with two hidden layers + dropout for improved performance.
    """

    def __init__(self, input_dim, hidden_dim1, hidden_dim2, num_classes):
        super(DeeperMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()

        self.dropout = nn.Dropout(p=0.3)

        self.fc3 = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.relu2(x)

        x = self.dropout(x)
        x = self.fc3(x)
        return x


def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X.float())
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    return acc, prec, rec


def main():
    print("\n=== Deeper Neural Network (MLP) with Dropout ===")

    train_dir = "data/face-expression-recognition-dataset/images/images/new_train"
    val_dir = "data/face-expression-recognition-dataset/images/images/new_validation"
    img_size = (48, 48)
    input_dim = img_size[0] * img_size[1]  # 48*48 = 2304
    hidden_dim1 = 256
    hidden_dim2 = 128
    batch_size = 64
    num_epochs = 15
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ====================================
    # 1) Fit a scaler to the train data
    # ====================================
    # We'll load the entire train set in memory to compute a standard scaler
    train_imgs = []
    for c in os.listdir(train_dir):
        c_path = os.path.join(train_dir, c)
        if not os.path.isdir(c_path):
            continue
        for f in os.listdir(c_path):
            img_path = os.path.join(c_path, f)
            if os.path.isfile(img_path):
                train_imgs.append(img_path)

    # Build a temporary array to fit scaler
    train_array = []
    for img_path in train_imgs:
        with Image.open(img_path).convert('L') as img:
            img = img.resize(img_size)
            arr = np.array(img, dtype=np.float32).reshape(-1)
            train_array.append(arr)
    train_array = np.array(train_array)

    scaler = StandardScaler()
    scaler.fit(train_array)

    # ====================================
    # 2) Create dataset & dataloaders
    # ====================================
    train_dataset = ExpressionDatasetMLP(train_dir, img_size=img_size, scaler=scaler)
    val_dataset = ExpressionDatasetMLP(val_dir, img_size=img_size, scaler=scaler)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    num_classes = len(train_dataset.class_to_idx)

    # ====================================
    # 3) Model & Optimizer
    # ====================================
    model = DeeperMLP(input_dim, hidden_dim1, hidden_dim2, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Optional: a simple step scheduler
    # step_size=5 => every 5 epochs, gamma=0.5 => half the LR
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    print("Starting MLP training...\n")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X.float())
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # LR scheduler step
        scheduler.step()

        # Evaluate
        val_acc, val_prec, val_rec = evaluate_model(model, val_loader, device)
        print(f"Epoch [{epoch + 1}/{num_epochs}] "
              f"Train_Loss: {running_loss / len(train_loader):.4f} | "
              f"Val_Acc: {val_acc:.4f} | "
              f"Val_Prec: {val_prec:.4f} | "
              f"Val_Rec: {val_rec:.4f}")

    # Final metrics
    final_acc, final_prec, final_rec = evaluate_model(model, val_loader, device)
    print("\n=== Final Evaluation on Validation Set ===")
    print(f"Accuracy:  {final_acc:.4f}")
    print(f"Precision: {final_prec:.4f}")
    print(f"Recall:    {final_rec:.4f}")

    # Save results
    with open("basic_mlp_results.txt", "w") as f:
        f.write(f"{final_acc:.4f},{final_prec:.4f},{final_rec:.4f}")

    print("\nDeeper MLP training complete! Results saved to 'basic_mlp_results.txt'.")


if __name__ == "__main__":
    main()
