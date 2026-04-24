"""
MPA-MLF Final Project — Room Occupancy Classification
60 GHz Delay-Doppler domain image classification (4 classes: 0, 1, 2, 3 persons)

Authors  : El Houari Arbouz + Amine Zihoune


Strategy overview:
  1. EfficientNet-B4 backbone (19M params) pre-trained on ImageNet
  2. 5-Fold Stratified Cross-Validation -> 5 independent models
  3. Input resolution 160x160 pixels for richer spatial features
  4. Mixup + CutMix augmentation (alternated randomly each batch)
  5. Pseudo-labeling: re-train on high-confidence test predictions
  6. Test-Time Augmentation (TTA) x32 passes at inference
  7. OneCycleLR scheduler for faster and deeper convergence
  8. 80 epochs per fold
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset, ConcatDataset
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
TRAIN_IMG_DIR   = "x_train"            # folder containing training PNG images
TEST_IMG_DIR    = "x_test"             # folder containing test PNG images
TRAIN_LABEL_CSV = "y_train_v2.csv"     # ground truth labels (0-based index)
SUBMISSION_CSV  = "output/submission.csv"  # Kaggle submission output

IMG_SIZE        = 160    # input resolution — B4 benefits from higher resolution
BATCH_SIZE      = 24     # reduced to fit B4 + 160px in memory
NUM_EPOCHS      = 80     # more epochs allow deeper convergence
LR              = 2e-4   # base learning rate for AdamW
NUM_CLASSES     = 4      # 0P / 1P / 2P / 3P
N_FOLDS         = 5      # number of cross-validation folds
TTA_STEPS       = 32     # number of TTA passes at inference
PSEUDO_EPOCHS   = 40     # epochs for pseudo-label fine-tuning phase
PSEUDO_CONF     = 0.92   # minimum softmax confidence to accept a pseudo-label
SEED            = 42

# ─────────────────────────────────────────────
#  DEVICE SELECTION
# ─────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("[INFO] Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("[INFO] Using CUDA GPU")
else:
    DEVICE = torch.device("cpu")
    print("[WARNING] No GPU found — training on CPU (slow)")

torch.manual_seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────
class OccupancyDataset(Dataset):
    """
    Custom Dataset for delay-Doppler PNG images.

    NOTE: image files are numbered from 1 (e.g. img_1.png),
    while labels in the CSV are 0-based. The _num() function
    handles both naming formats: 'img_42.png' and '42.png'.
    """

    def __init__(self, img_dir, labels=None, transform=None, filenames=None):
        self.img_dir   = img_dir
        self.labels    = labels     # None for the test set
        self.transform = transform

        if filenames is not None:
            # use pre-provided filename list (e.g. for pseudo-labeling subset)
            self.filenames = filenames
        else:
            def _num(f):
                # extract the numeric ID from filenames like 'img_4702.png' or '4702.png'
                return int(os.path.splitext(f)[0].split("_")[-1])

            self.filenames = sorted(
                [f for f in os.listdir(img_dir) if f.endswith(".png")],
                key=_num
            )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_dir, self.filenames[idx])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.labels is not None:
            return img, int(self.labels[idx])
        return img


# ─────────────────────────────────────────────
#  DATA TRANSFORMS
# ─────────────────────────────────────────────
# ImageNet normalisation statistics (required for pre-trained EfficientNet)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# Training: moderate augmentation
# NOTE: augmentations are kept conservative because the spatial position
# of intensity peaks in the delay-Doppler domain carries physical meaning
# (distance and velocity). Aggressive transforms could corrupt this information.
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.4),
    transforms.RandomVerticalFlip(p=0.4),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.2),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.15, scale=(0.01, 0.08)),  # must come after ToTensor
])

# Validation / test: no augmentation
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# TTA: light random augmentation applied at inference time
tta_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# ─────────────────────────────────────────────
#  MIXUP AUGMENTATION
#  Linearly interpolates two images and their
#  labels. Forces the model to learn smoother
#  decision boundaries.
# ─────────────────────────────────────────────
def mixup_batch(imgs, labels, alpha=0.3):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(imgs.size(0), device=imgs.device)
    mixed = lam * imgs + (1 - lam) * imgs[idx]
    return mixed, labels, labels[idx], lam


# ─────────────────────────────────────────────
#  CUTMIX AUGMENTATION
#  Cuts a random patch from one image and
#  pastes it onto another. The mixed label is
#  weighted by the area ratio of the patch.
# ─────────────────────────────────────────────
def cutmix_batch(imgs, labels, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(imgs.size(0), device=imgs.device)
    B, C, H, W = imgs.shape

    # compute patch size proportional to lambda
    cut_rat = np.sqrt(1.0 - lam)
    cut_w   = int(W * cut_rat)
    cut_h   = int(H * cut_rat)

    # random patch centre
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    mixed = imgs.clone()
    mixed[:, :, y1:y2, x1:x2] = imgs[idx, :, y1:y2, x1:x2]

    # recompute lambda based on actual patch area
    lam_real = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return mixed, labels, labels[idx], lam_real


# ─────────────────────────────────────────────
#  MODEL — EfficientNet-B4 + Custom Head
# ─────────────────────────────────────────────
def build_model():
    """
    Load EfficientNet-B4 pre-trained on ImageNet and replace the
    classification head with a deeper head suited for 4-class output.

    EfficientNet-B4 features:
      - 19M parameters (vs 5.3M for B0)
      - 1792-dim feature vector before the classifier
      - compound scaling: wider, deeper, and higher resolution than B0
    """
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features  # 1792 for B4

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.3),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(128, NUM_CLASSES),
    )
    return model


# ─────────────────────────────────────────────
#  LABEL LOADING
# ─────────────────────────────────────────────
def load_labels(csv_path):
    """Load ground truth labels from CSV, sorted by the ID column."""
    df = pd.read_csv(csv_path)
    df = df.sort_values(df.columns[0]).reset_index(drop=True)
    return df[df.columns[1]].values


# ─────────────────────────────────────────────
#  TRAINING — ONE EPOCH
#  Randomly applies one of three strategies
#  per batch: Mixup, CutMix, or standard CE.
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, n = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        r = np.random.rand()
        if r < 0.33:
            # --- Mixup (33% of batches) ---
            imgs, ya, yb, lam = mixup_batch(imgs, labels)
            out  = model(imgs)
            loss = lam * criterion(out, ya) + (1 - lam) * criterion(out, yb)
        elif r < 0.66:
            # --- CutMix (33% of batches) ---
            imgs, ya, yb, lam = cutmix_batch(imgs, labels)
            out  = model(imgs)
            loss = lam * criterion(out, ya) + (1 - lam) * criterion(out, yb)
        else:
            # --- Standard cross-entropy (34% of batches) ---
            out  = model(imgs)
            loss = criterion(out, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        n          += imgs.size(0)

    return total_loss / n, correct / n


# ─────────────────────────────────────────────
#  EVALUATION — ONE EPOCH
# ─────────────────────────────────────────────
@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_preds, all_labels  = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out  = model(imgs)
        loss = criterion(out, labels)

        total_loss += loss.item() * imgs.size(0)
        preds = out.argmax(1)
        correct += (preds == labels).sum().item()
        n       += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / n, correct / n, all_preds, all_labels


# ─────────────────────────────────────────────
#  TEST-TIME AUGMENTATION (TTA)
#  Run inference N times with random augmentation
#  and average the softmax probabilities.
#  The first pass always uses the clean val_transform.
# ─────────────────────────────────────────────
@torch.no_grad()
def tta_probabilities(model, img_dir, filenames, n_tta=TTA_STEPS):
    """
    Returns accumulated softmax probability matrix of shape (N_images, 4).
    Summing across n_tta passes — caller divides by ensemble size if needed.
    """
    model.eval()
    accumulated = None

    for i in range(n_tta):
        # first pass: clean image; subsequent passes: randomly augmented
        t = val_transform if i == 0 else tta_transform
        ds = OccupancyDataset(img_dir, labels=None, transform=t, filenames=filenames)
        loader = DataLoader(ds, batch_size=48, shuffle=False, num_workers=0)

        probs_list = []
        for imgs in loader:
            logits = model(imgs.to(DEVICE))
            probs_list.append(torch.softmax(logits, dim=1).cpu().numpy())

        probs = np.concatenate(probs_list, axis=0)  # shape: (N, 4)
        accumulated = probs if accumulated is None else accumulated + probs

    return accumulated


# ─────────────────────────────────────────────
#  TRAINING — ONE FOLD
# ─────────────────────────────────────────────
def train_fold(fold_idx, idx_train, idx_val, labels,
               n_epochs=NUM_EPOCHS, extra_dataset=None):
    """
    Train one fold of the cross-validation.

    Args:
        fold_idx      : current fold index (0-based)
        idx_train     : indices of training samples for this fold
        idx_val       : indices of validation samples for this fold
        labels        : full label array
        n_epochs      : number of epochs to train
        extra_dataset : optional pseudo-labeled dataset to append to training set

    Returns:
        model     : best model (loaded from checkpoint)
        best_acc  : best validation accuracy achieved
        history   : dict with train_acc and val_acc lists
    """
    print(f"\n{'─'*60}")
    print(f"  FOLD {fold_idx + 1}/{N_FOLDS}  |  {n_epochs} epochs")
    print(f"{'─'*60}")

    # build datasets
    base_train = OccupancyDataset(TRAIN_IMG_DIR, labels, train_transform)
    train_sub  = Subset(base_train, idx_train)
    val_set    = Subset(OccupancyDataset(TRAIN_IMG_DIR, labels, val_transform), idx_val)

    # optionally append pseudo-labeled test images
    if extra_dataset is not None:
        train_sub = ConcatDataset([train_sub, extra_dataset])
        print(f"  [Pseudo-label] {len(extra_dataset)} extra samples added to training set")

    # WeightedRandomSampler: over-sample minority classes to compensate imbalance
    train_labels_sub = labels[idx_train]
    class_weights    = 1.0 / np.bincount(train_labels_sub)
    sample_weights   = class_weights[train_labels_sub]
    sampler = WeightedRandomSampler(
        torch.DoubleTensor(sample_weights), len(sample_weights), replacement=True
    )

    train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,   num_workers=0)

    model     = build_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # OneCycleLR: ramps up LR then decays — faster convergence than cosine annealing
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR * 10,
        steps_per_epoch=len(train_loader),
        epochs=n_epochs,
        pct_start=0.1
    )

    best_acc   = 0.0
    model_path = f"best_model_fold{fold_idx + 1}.pth"
    history    = {"train_acc": [], "val_acc": []}

    for epoch in range(1, n_epochs + 1):
        tr_loss, tr_acc       = train_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_acc, _, _ = eval_epoch(model, val_loader, criterion)
        scheduler.step()

        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        # save checkpoint whenever validation accuracy improves
        if vl_acc > best_acc:
            best_acc = vl_acc
            torch.save(model.state_dict(), model_path)

        if epoch % 10 == 0 or epoch == 1:
            tag = " [BEST]" if vl_acc == best_acc else ""
            print(f"  Epoch {epoch:3d}/{n_epochs} | "
                  f"Train {tr_acc:.4f} | Val {vl_acc:.4f}{tag}")

    print(f"  Fold {fold_idx + 1} best validation accuracy: {best_acc:.4f}")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    return model, best_acc, history


# ─────────────────────────────────────────────
#  PSEUDO-LABELING
#
#  Step 1: predict test set with the Phase 1 ensemble
#  Step 2: keep only highly confident predictions (>= PSEUDO_CONF)
#  Step 3: add them to the training set and fine-tune
#
#  Rationale: the test images share the same distribution as
#  the training data. Adding confident pseudo-labeled test
#  samples effectively increases the training set size and
#  helps the model adapt to test-specific variations.
# ─────────────────────────────────────────────
def build_pseudo_dataset(models_list, test_ds):
    print(f"\n{'='*60}")
    print("  PSEUDO-LABELING — generating pseudo-labels from test set...")
    print(f"{'='*60}")

    filenames      = test_ds.filenames
    ensemble_probs = None

    # ensemble prediction with 8 TTA passes (faster than full TTA for pseudo-labels)
    for i, model in enumerate(models_list):
        probs = tta_probabilities(model, TEST_IMG_DIR, filenames, n_tta=8)
        ensemble_probs = probs if ensemble_probs is None else ensemble_probs + probs

    avg_probs    = ensemble_probs / len(models_list)
    max_probs    = avg_probs.max(axis=1)
    pseudo_preds = avg_probs.argmax(axis=1)

    # filter by confidence threshold
    confident_mask = max_probs >= PSEUDO_CONF
    n_confident    = confident_mask.sum()

    print(f"  Total test images         : {len(filenames)}")
    print(f"  Pseudo-labels accepted    : {n_confident} ({100 * n_confident / len(filenames):.1f}%)")
    print(f"  Confidence threshold      : {PSEUDO_CONF}")
    print(f"  Class distribution        : {np.bincount(pseudo_preds[confident_mask])}")

    if n_confident == 0:
        print("  [WARNING] No pseudo-labels above confidence threshold — skipping this step")
        return None

    confident_fns    = [filenames[i] for i in range(len(filenames)) if confident_mask[i]]
    confident_labels = pseudo_preds[confident_mask]

    pseudo_ds = OccupancyDataset(
        TEST_IMG_DIR,
        labels=confident_labels,
        transform=train_transform,
        filenames=confident_fns
    )
    return pseudo_ds


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  MPA-MLF — EfficientNet-B4 | 5-Fold | TTA x32 | Pseudo-Label")
    print("  Estimated runtime on Apple M4: ~7-8 hours")
    print("=" * 60 + "\n")

    os.makedirs("output", exist_ok=True)

    # ── Load labels ───────────────────────────
    labels = load_labels(TRAIN_LABEL_CSV)
    print(f"[INFO] {len(labels)} training images loaded")
    print(f"       Class distribution: {np.bincount(labels)}  (0P / 1P / 2P / 3P)\n")

    # Stratified K-Fold split
    skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.arange(len(labels)), labels))

    # ══════════════════════════════════════════
    #  PHASE 1 — Initial 5-Fold Training
    # ══════════════════════════════════════════
    print("=" * 60)
    print(f"  PHASE 1 — Initial training ({N_FOLDS} folds x {NUM_EPOCHS} epochs)")
    print("=" * 60)

    models_list   = []
    all_val_accs  = []
    all_histories = []

    for fold_idx, (idx_train, idx_val) in enumerate(splits):
        model, best_acc, history = train_fold(fold_idx, idx_train, idx_val, labels)
        models_list.append(model)
        all_val_accs.append(best_acc)
        all_histories.append(history)

    print(f"\n{'='*60}")
    print(f"  Phase 1 complete")
    print(f"  Accuracy per fold : {[f'{a:.4f}' for a in all_val_accs]}")
    print(f"  Mean accuracy     : {np.mean(all_val_accs):.4f}")
    print(f"{'='*60}\n")

    # ══════════════════════════════════════════
    #  PHASE 2 — Pseudo-Labeling + Fine-Tuning
    # ══════════════════════════════════════════
    print("=" * 60)
    print("  PHASE 2 — Pseudo-labeling + fine-tuning")
    print("=" * 60)

    test_ds   = OccupancyDataset(TEST_IMG_DIR, labels=None, transform=val_transform)
    pseudo_ds = build_pseudo_dataset(models_list, test_ds)

    models_v2   = []
    all_accs_v2 = []

    for fold_idx, (idx_train, idx_val) in enumerate(splits):
        model, best_acc, history = train_fold(
            fold_idx, idx_train, idx_val, labels,
            n_epochs=PSEUDO_EPOCHS,
            extra_dataset=pseudo_ds
        )
        models_v2.append(model)
        all_accs_v2.append(best_acc)
        all_histories.append(history)

    print(f"\n{'='*60}")
    print(f"  Phase 2 complete (with pseudo-labels)")
    print(f"  Accuracy per fold : {[f'{a:.4f}' for a in all_accs_v2]}")
    print(f"  Mean accuracy     : {np.mean(all_accs_v2):.4f}")
    print(f"{'='*60}\n")

    # Select the best model per fold between Phase 1 and Phase 2
    final_models = []
    for i in range(N_FOLDS):
        if all_accs_v2[i] >= all_val_accs[i]:
            final_models.append(models_v2[i])
            print(f"  Fold {i+1}: using Phase 2 model ({all_accs_v2[i]:.4f} >= {all_val_accs[i]:.4f})")
        else:
            final_models.append(models_list[i])
            print(f"  Fold {i+1}: keeping Phase 1 model ({all_val_accs[i]:.4f} > {all_accs_v2[i]:.4f})")

    # ── Learning curves plot ──────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    for i in range(N_FOLDS):
        ax.plot(all_histories[i]["val_acc"], label=f"Fold {i + 1} — val")
    ax.set_title("Validation Accuracy per Fold — Phase 1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    plt.tight_layout()
    plt.savefig("output/learning_curves.png", dpi=150)
    print("\n[INFO] Learning curves saved to output/learning_curves.png")

    # ══════════════════════════════════════════
    #  PHASE 3 — Final Predictions (TTA x32 Ensemble)
    # ══════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  PHASE 3 — Final predictions (TTA x{TTA_STEPS} x {N_FOLDS} models)")
    print(f"{'='*60}\n")

    test_filenames = test_ds.filenames
    ensemble_probs = None

    for i, model in enumerate(final_models):
        print(f"  Running TTA for model {i + 1}/{N_FOLDS}...")
        probs = tta_probabilities(model, TEST_IMG_DIR, test_filenames, n_tta=TTA_STEPS)
        ensemble_probs = probs if ensemble_probs is None else ensemble_probs + probs

    # majority vote via accumulated softmax probabilities
    final_preds = ensemble_probs.argmax(axis=1)

    # ── Confusion matrix (last fold validation set) ──
    _, _, val_preds, val_true = eval_epoch(
        final_models[-1],
        DataLoader(
            Subset(OccupancyDataset(TRAIN_IMG_DIR, labels, val_transform), splits[-1][1]),
            batch_size=64, shuffle=False, num_workers=0
        ),
        nn.CrossEntropyLoss()
    )
    cm = confusion_matrix(val_true, val_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["0P", "1P", "2P", "3P"],
                yticklabels=["0P", "1P", "2P", "3P"])
    plt.title("Confusion Matrix — Validation set (Fold 5)")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig("output/confusion_matrix.png", dpi=150)
    print("[INFO] Confusion matrix saved to output/confusion_matrix.png")

    # ── Export Kaggle submission CSV ──────────
    def _fname_to_id(fname):
        # convert filename to 0-based Kaggle ID
        # e.g. 'img_4702.png' -> 4701
        return int(os.path.splitext(fname)[0].split("_")[-1]) - 1

    test_ids   = [_fname_to_id(f) for f in test_filenames]
    submission = pd.DataFrame({"Id": test_ids, "label": final_preds})
    submission = submission.sort_values("Id").reset_index(drop=True)
    submission.to_csv(SUBMISSION_CSV, index=False)

    print(f"\n[INFO] Submission saved to {SUBMISSION_CSV}")
    print(f"       Predicted class distribution: {np.bincount(final_preds)}")
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  Phase 1 mean val accuracy : {np.mean(all_val_accs):.4f}")
    print(f"  Phase 2 mean val accuracy : {np.mean(all_accs_v2):.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
