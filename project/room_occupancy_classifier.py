"""
MPA-MLF Final Project — Room Occupancy Classification
60 GHz Delay-Doppler domain image classification (4 classes: 0, 1, 2, 3 persons)

Author  : El Houari ARBOUZ + Amine ZIHOUNE

"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns


# ─────────────────────────────────────────────
#  CONFIG 
# ─────────────────────────────────────────────
TRAIN_IMG_DIR   = "x_train"                    # folder containing training .png files
TEST_IMG_DIR    = "x_test"                     # folder containing test .png files
TRAIN_LABEL_CSV = "y_train_v2.csv"             # training labels
SUBMISSION_CSV  = "output/submission.csv"      # output file for Kaggle

IMG_SIZE    = 64         # delay-Doppler images are small; 64 is enough
BATCH_SIZE  = 64
NUM_EPOCHS  = 60
LR          = 1e-3
NUM_CLASSES = 4
SEED        = 42

# ─────────────────────────────────────────────
#  DEVICE — MPS for M4, otherwise CPU
# ─────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("✅  Using Apple Silicon GPU (MPS)")
else:
    DEVICE = torch.device("cpu")
    print("⚠️  Using CPU — slower training")

torch.manual_seed(SEED)
np.random.seed(SEED)


    # ─────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────
class OccupancyDataset(Dataset):
    """
    Loads PNG images from a folder.
    WARNING: images are numbered starting from 1 (1.png, 2.png ...)
             labels are numbered from 0 in the CSV
    """
    def __init__(self, img_dir, labels=None, transform=None):
        self.img_dir   = img_dir
        self.labels    = labels       # None pour le jeu de test
        self.transform = transform

        # numerically sorted list — supports both "img_123.png" AND "123.png"
        def _parse_num(fname):
            stem = os.path.splitext(fname)[0]   # "img_4702" -> prend la partie après "_"
            return int(stem.split("_")[-1])

        self.filenames = sorted(
            [f for f in os.listdir(img_dir) if f.endswith(".png")],
            key=_parse_num
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.filenames[idx])
        image    = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.labels is not None:
            label = int(self.labels[idx])
            return image, label
        return image


 # ─────────────────────────────────────────────
#  TRANSFORMATIONS (augmentation)
# ─────────────────────────────────────────────
# Light augmentation: the spots in the image have physical meaning,
# so we avoid aggressive horizontal/vertical flips.
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


 # ─────────────────────────────────────────────
#  MODEL — Fine-tuned EfficientNet-B0
# ─────────────────────────────────────────────
def build_model(num_classes=4):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Replace the final classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )
    return model


 # ─────────────────────────────────────────────
#  LABEL LOADING
# ─────────────────────────────────────────────
def load_labels(csv_path):
    df = pd.read_csv(csv_path)
    # Assumes columns: Id, label (or similar)
    # Sorts by Id to match the order of images
    id_col    = df.columns[0]
    label_col = df.columns[1]
    df = df.sort_values(by=id_col).reset_index(drop=True)
    return df[label_col].values


 # ─────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss, correct = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        # Mixed precision (not natively supported on MPS but has no negative effect)
        with torch.autocast(device_type="cpu"):  # cpu for MPS compatibility
            outputs = model(imgs)
            loss    = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()

    n = len(loader.dataset)
    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct = 0, 0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        loss    = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    n = len(loader.dataset)
    return total_loss / n, correct / n, all_preds, all_labels


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    print("\n" + "="*55)
    print("  MPA-MLF — Room Occupancy Classification")
    print("="*55 + "\n")

    # --- Load labels ---
    labels = load_labels(TRAIN_LABEL_CSV)
    print(f"📂  {len(labels)} training samples found")
    print(f"    Class distribution: {np.bincount(labels)}\n")

    # --- Split train / validation (80 / 20) ---
    from sklearn.model_selection import train_test_split
    indices     = np.arange(len(labels))
    idx_train, idx_val = train_test_split(
        indices, test_size=0.20, stratify=labels, random_state=SEED
    )

    full_dataset = OccupancyDataset(TRAIN_IMG_DIR, labels=labels, transform=None)

    # Subsets
    from torch.utils.data import Subset
    train_set = Subset(OccupancyDataset(TRAIN_IMG_DIR, labels=labels, transform=train_transform), idx_train)
    val_set   = Subset(OccupancyDataset(TRAIN_IMG_DIR, labels=labels, transform=val_transform),   idx_val)

    # Weighted sampler to balance classes if there is imbalance
    train_labels_sub = labels[idx_train]
    class_counts     = np.bincount(train_labels_sub)
    class_weights    = 1.0 / class_counts
    sample_weights   = class_weights[train_labels_sub]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(train_labels_sub),
        replacement=True
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,  num_workers=0, pin_memory=False)

    # --- Model ---
    model     = build_model(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # Scheduler: CosineAnnealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
    scaler    = None  # GradScaler not used on MPS

    # --- Training loop ---
    best_val_acc  = 0.0
    best_model_path = "best_model.pth"
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"🚀  Starting training on {DEVICE} | {NUM_EPOCHS} epochs\n")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, scaler)
        val_loss, val_acc, preds, true = eval_epoch(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        marker = "  ✅ BEST" if val_acc > best_val_acc else ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{NUM_EPOCHS} | "
                  f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
                  f"Val loss: {val_loss:.4f} acc: {val_acc:.4f}{marker}")

    print(f"\n🏆  Best validation accuracy: {best_val_acc:.4f}")

    # --- Learning curves ---
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(history["train_loss"], label="Train")
    axs[0].plot(history["val_loss"],   label="Val")
    axs[0].set_title("Loss"); axs[0].legend()
    axs[1].plot(history["train_acc"], label="Train")
    axs[1].plot(history["val_acc"],   label="Val")
    axs[1].set_title("Accuracy"); axs[1].legend()
    plt.tight_layout()
    plt.savefig("learning_curves.png", dpi=150)
    print("📊  Curves saved: learning_curves.png")

    # --- Confusion matrix on validation set ---
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    _, _, val_preds, val_true = eval_epoch(model, val_loader, criterion)
    cm = confusion_matrix(val_true, val_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["0P","1P","2P","3P"],
                yticklabels=["0P","1P","2P","3P"])
    plt.title("Confusion Matrix — Validation set")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("📊  Confusion matrix saved: confusion_matrix.png")

    # ─────────────────────────────────────────
    #  PREDICTIONS ON TEST SET
    # ─────────────────────────────────────────
    print("\n🔮  Generating predictions on x_test...")
    test_dataset = OccupancyDataset(TEST_IMG_DIR, labels=None, transform=val_transform)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model.eval()
    all_preds = []
    with torch.no_grad():
        for imgs in tqdm(test_loader, desc="Prediction"):
            imgs    = imgs.to(DEVICE)
            outputs = model(imgs)
            preds   = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)

    # Kaggle format: Id (0-based), label
    test_filenames = test_dataset.filenames
    # extract the number from "img_XXXX.png" or "XXXX.png", then -1 to switch to 0-based
    def _fname_to_id(fname):
        stem = os.path.splitext(fname)[0]
        return int(stem.split("_")[-1]) - 1

    test_ids = [_fname_to_id(f) for f in test_filenames]

    os.makedirs("output", exist_ok=True)
    submission = pd.DataFrame({"Id": test_ids, "label": all_preds})
    submission = submission.sort_values("Id").reset_index(drop=True)
    submission.to_csv(SUBMISSION_CSV, index=False)
    print(f"✅  Submission saved: {SUBMISSION_CSV}")
    print(f"    Predictions distribution: {np.bincount(all_preds)}\n")

    print("="*55)
    print("  ✅  Pipeline successfully completed!")
    print(f"     Val accuracy : {best_val_acc:.4f}")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
