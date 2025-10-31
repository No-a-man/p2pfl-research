from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
import time
import matplotlib.pyplot as plt
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from simpleTextSVM import SimpleTextSVM  # ✅ changed model import

# 1️⃣ Load MNIST dataset (same as P2PFL)
print("📥 Loading MNIST dataset (same as P2PFL)...")
start_time = time.time()
data = P2PFLDataset.from_huggingface("p2pfl/MNIST")
print(f"📊 Dataset loaded from P2PFL")

# Extract data from P2PFL dataset
X_data = []
y_data = []
sample_count = 0

try:
    if hasattr(data, 'data'):
        dataset_data = data.data
        print(f"📊 Found dataset.data with {len(dataset_data)} samples")
        for i, sample in enumerate(dataset_data):
            if i >= 70000:
                break
            X_data.append(sample["image"].flatten().numpy())
            y_data.append(sample["label"].item())
            sample_count += 1
    elif hasattr(data, '__iter__'):
        print("📊 Dataset is iterable, trying to iterate...")
        for i, sample in enumerate(data):
            if i >= 70000:
                break
            X_data.append(sample["image"].flatten().numpy())
            y_data.append(sample["label"].item())
            sample_count += 1
    else:
        print("⚠️  Cannot access dataset data directly")
        print("🔄 Creating dummy data for testing...")
        np.random.seed(42)
        X_data = [np.random.randn(784) for _ in range(70000)]
        y_data = [np.random.randint(0, 10) for _ in range(70000)]
        sample_count = 70000
except Exception as e:
    print(f"⚠️  Error accessing dataset: {e}")
    print("🔄 Creating dummy data for testing...")
    np.random.seed(42)
    X_data = [np.random.randn(784) for _ in range(70000)]
    y_data = [np.random.randint(0, 10) for _ in range(70000)]
    sample_count = 70000

print(f"📊 Successfully loaded {sample_count} samples")
X = np.array(X_data)
y = np.array(y_data)
print(f"📊 Using {X.shape[0]} samples (same as P2PFL)")

# 2️⃣ Convert labels → binary (1 if '0', 0 otherwise)
print(f"📈 Data shape: X={X.shape}, y={y.shape}")
print(f"🏷️  Classes: {np.unique(y)}")
print(f"📊 Class distribution: {np.bincount(y)}")

y_binary = np.where(y == 0, 1, 0)
print(f"🔄 Converted to binary classification: 0=1 (positive), others=0 (negative)")
print(f"📈 Binary distribution: {np.bincount(y_binary)}")
print(f"📊 Class distribution: Non-zero digits: {np.sum(y_binary == 0)}, Zero digits: {np.sum(y_binary == 1)}")

# 3️⃣ Normalize pixel values
print("🔄 Standardizing features...")
data_loading_time = time.time() - start_time
print(f"⏱️  Data loading time: {data_loading_time:.2f} seconds")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("✅ Standardization complete")

# We'll train using the numeric features directly with a small PyTorch training loop
# Split using the original digit labels (SimpleTextSVM converts to binary internally)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

import torch
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 128
# SimpleTextSVM.forward expects input shaped (batch, 1, input_size)
train_tensor_x = torch.tensor(X_train, dtype=torch.float32).view(-1, 1, X_train.shape[1])
train_tensor_y = torch.tensor(y_train, dtype=torch.long)
test_tensor_x = torch.tensor(X_test, dtype=torch.float32).view(-1, 1, X_test.shape[1])
test_tensor_y = torch.tensor(y_test, dtype=torch.long)

train_ds = TensorDataset(train_tensor_x, train_tensor_y)
test_ds = TensorDataset(test_tensor_x, test_tensor_y)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# 5️⃣ Train SimpleTextSVM with explicit loop and record loss per epoch
print("🚀 Training SimpleTextSVM (PyTorch loop)...")
training_start_time = time.time()
svm = SimpleTextSVM(input_size=X_train.shape[1], lr_rate=0.01, C=1.0)
svm.to(device)
optimizer = torch.optim.SGD(svm.parameters(), lr=svm.lr_rate)

epochs = 10
loss_history = []
for epoch in range(1, epochs + 1):
    svm.train()
    running_loss = 0.0
    batch_count = 0
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        outputs = svm(xb)
        loss = svm.hinge_loss(outputs, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        batch_count += 1
    avg_loss = running_loss / max(1, batch_count)
    loss_history.append(avg_loss)
    print(f"Epoch {epoch}/{epochs} - loss: {avg_loss:.6f}")

training_end_time = time.time()
training_time = training_end_time - training_start_time
print("✅ Training complete")
print(f"⏱️  Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

# Plot loss vs epoch
try:
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), loss_history, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Epoch (Centralized Text SVM)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_vs_epoch_centralizedTextSVM.png')
    plt.show()
    print("📈 Saved loss plot to loss_vs_epoch_centralizedTextSVM.png")
except Exception as e:
    print(f"⚠️  Could not plot/save loss: {e}")

# 6️⃣ Evaluate (PyTorch inference)
print("🔍 Evaluating model...")
evaluation_start_time = time.time()
svm.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        logits = svm(xb)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(yb.numpy())

if len(all_preds) > 0:
    y_pred = np.concatenate(all_preds)
    y_true_digits = np.concatenate(all_labels)
else:
    y_pred = np.array([])
    y_true_digits = np.array([])

# Convert ground-truth digits to the binary scheme used earlier: 0 -> 1 (positive), others -> 0
y_test_binary = np.where(y_true_digits == 0, 1, 0)

evaluation_end_time = time.time()
evaluation_time = evaluation_end_time - evaluation_start_time

total_experiment_time = time.time() - start_time
print(f"✅ Evaluation complete")
print(f"⏱️  Evaluation time: {evaluation_time:.2f} seconds")

# Metrics (use binary labels)
precision = precision_score(y_test_binary, y_pred)
recall = recall_score(y_test_binary, y_pred)
f1 = f1_score(y_test_binary, y_pred)
cm = confusion_matrix(y_test_binary, y_pred)
# Ensure confusion matrix is 2x2
if cm.size == 4:
    tn, fp, fn, tp = cm.ravel()
else:
    # handle degenerate cases
    tn = fp = fn = tp = 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
balanced_accuracy = (sensitivity + specificity) / 2

print(f"\n🎯 CENTRALIZED TEXT-SVM RESULTS:")
print(f"=" * 60)
print(f"📊 Problem: Digit 0 (positive) vs Non-zero digits (1–9) (negative)")
print(f"🎯 Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print(f"📈 Precision: {precision:.4f}")
print(f"📈 Recall: {recall:.4f}")
print(f"📈 Specificity: {specificity:.4f}")
print(f"📈 F1-Score: {f1:.4f}")
print(f"📈 Balanced Accuracy: {balanced_accuracy:.4f}")

print(f"\n📊 CONFUSION MATRIX:\n{cm}")
print(f"\n📊 CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=['Non-zero', 'Zero']))

# Timings
print(f"\n📊 TIMING SUMMARY:")
print(f"⏱️  Data loading time: {data_loading_time:.2f} seconds")
print(f"⏱️  Training time: {training_time:.2f} seconds")
print(f"⏱️  Evaluation time: {evaluation_time:.2f} seconds")
print(f"⏱️  Total experiment time: {total_experiment_time:.2f} seconds ({total_experiment_time/60:.2f} minutes)")
