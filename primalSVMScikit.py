import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset

# -----------------------------
# 1. Load & Preprocess Dataset (Same as P2PFL)
# -----------------------------
print("🚀 Loading dataset (same as P2PFL code)...")
start_time = time.time()

# Use the same dataset as P2PFL code
data = P2PFLDataset.from_huggingface("p2pfl/MNIST")
print(f"📊 Dataset loaded from P2PFL")

# Convert to numpy arrays for sklearn
X_data = []
y_data = []
sample_count = 0

# Iterate through the dataset to get all samples
try:
    i = 0
    while True:
        try:
            sample = data[i]
            X_data.append(sample["image"].flatten().numpy())  # Flatten 28x28 to 784
            y_data.append(sample["label"].item())
            sample_count += 1
            i += 1
        except (IndexError, KeyError):
            break
except Exception as e:
    print(f"⚠️  Error accessing dataset: {e}")
    # Fallback: use a subset of data
    print("🔄 Using fallback data loading...")
    for i in range(1000):  # Use first 1000 samples as fallback
        try:
            sample = data[i]
            X_data.append(sample["image"].flatten().numpy())
            y_data.append(sample["label"].item())
            sample_count += 1
        except:
            break

print(f"📊 Successfully loaded {sample_count} samples")

X = np.array(X_data)
y = np.array(y_data)

# Convert to binary classification: 0=1 (positive), others=0 (negative)
y_binary = (y == 0).astype(int)
print(f"🔄 Converted to binary classification: 0=1 (positive), others=0 (negative)")
print(f"📈 Binary distribution: {np.bincount(y_binary)}")
print(f"📊 Class distribution: Non-zero digits: {np.sum(y_binary == 0)}, Zero digits: {np.sum(y_binary == 1)}")

print(f"📈 Data shape: X={X.shape}, y={y.shape}")
print(f"🏷️  Classes: {np.unique(y)}")

# Split data into train/test (same split as P2PFL would use)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"✅ Dataset prepared: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
print(f"⏱️  Data loading time: {time.time() - start_time:.2f} seconds")

# -----------------------------
# 2. Define SGD-based Linear SVM (binary)
# -----------------------------
class BinarySVM:
    def __init__(self, lr=0.001, C=1.0, n_iters=50):
        self.lr = lr
        self.C = C
        self.n_iters = n_iters

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        self.loss_history = []
        self.time_history = []
        self.accuracy_history = []

        print(f"  🔄 Training binary SVM: {n_samples} samples, {n_features} features")
        
        for i in range(self.n_iters):
            iter_start = time.time()

            # Forward pass
            margin = y * (np.dot(X, self.w) + self.b)
            misclassified = margin < 1

            # Gradient computation
            dw = self.w - self.C * np.dot(X[misclassified].T, y[misclassified]) / n_samples
            db = -self.C * np.sum(y[misclassified]) / n_samples

            # Update parameters
            self.w -= self.lr * dw
            self.b -= self.lr * db

            # Compute loss and accuracy
            loss = 0.5 * np.dot(self.w, self.w) + self.C * np.mean(np.maximum(0, 1 - margin))
            predictions = np.sign(np.dot(X, self.w) + self.b)
            accuracy = np.mean(predictions == y)
            
            self.loss_history.append(loss)
            self.accuracy_history.append(accuracy)
            self.time_history.append(time.time() - iter_start)
            
            if (i + 1) % 10 == 0:
                print(f"    Iter {i+1:2d}: Loss={loss:.4f}, Acc={accuracy:.4f}")

    def get_metrics(self):
        return {
            'final_loss': self.loss_history[-1],
            'final_accuracy': self.accuracy_history[-1],
            'avg_iter_time': np.mean(self.time_history),
            'total_time': np.sum(self.time_history)
        }

    def decision_function(self, X):
        return np.dot(X, self.w) + self.b

# -----------------------------
# 3. One-vs-Rest Wrapper for Multi-class
# -----------------------------
class BinarySVMClassifier:
    """Binary SVM for digit 0 vs non-zero classification"""
    
    def __init__(self, lr=0.001, C=1.0, n_iters=50):
        self.lr = lr
        self.C = C
        self.n_iters = n_iters

    def fit(self, X, y):
        print(f"\n🎯 Training Binary SVM (Digit 0 vs Non-Zero)...")
        print(f"📊 Training on {X.shape[0]} samples, {X.shape[1]} features")
        start_total = time.time()
        
        # Convert to binary labels: 0=1 (positive), others=0 (negative)
        y_binary = (y == 0).astype(int)
        
        # Train single binary classifier
        self.classifier = BinarySVM(lr=self.lr, C=self.C, n_iters=self.n_iters)
        self.classifier.fit(X, y_binary)
        
        self.total_train_time = time.time() - start_total
        print(f"\n✅ Binary SVM training completed in {self.total_train_time:.2f} seconds.")
        
        # Calculate metrics
        self.metrics = self.classifier.get_metrics()
        self.overall_metrics = {
            'total_train_time': self.total_train_time,
            'final_loss': self.metrics['final_loss'],
            'final_accuracy': self.metrics['final_accuracy']
        }

    def predict(self, X):
        """Predict binary labels: 0=non-zero digit, 1=zero digit"""
        decision_scores = self.classifier.decision_function(X)
        return (decision_scores > 0).astype(int)
    
    def get_convergence_data(self):
        """Get convergence data for plotting"""
        return {
            'loss_history': self.classifier.loss_history,
            'accuracy_history': self.classifier.accuracy_history
        }

# -----------------------------
# 4. Train and Evaluate
# -----------------------------
print(f"\n🚀 Starting Scikit-learn SVM Training...")
training_start = time.time()

model = BinarySVMClassifier(lr=0.001, C=1.0, n_iters=50)
model.fit(X_train, y_train)

# Test the model
print(f"\n🧪 Evaluating on test set...")
test_start = time.time()
y_pred = model.predict(X_test)
test_time = time.time() - test_start

# Calculate metrics
test_accuracy = accuracy_score(y_test, y_pred)
total_experiment_time = time.time() - training_start

# -----------------------------
# 5. Comprehensive Results & Visualization
# -----------------------------
print(f"\n📊 BINARY SVM RESULTS:")
print(f"=" * 50)
print(f"🎯 Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"⏱️  Total Training Time: {model.total_train_time:.2f} seconds")
print(f"⏱️  Test Evaluation Time: {test_time:.2f} seconds")
print(f"⏱️  Total Experiment Time: {total_experiment_time:.2f} seconds")
print(f"📉 Final Training Loss: {model.overall_metrics['final_loss']:.4f}")
print(f"📊 Final Training Accuracy: {model.overall_metrics['final_accuracy']:.4f}")

# Binary classification metrics
print(f"\n📋 Binary Classification Details:")
print(f"  🎯 Problem: Digit 0 (positive) vs Non-zero digits (1,2,3,4,5,6,7,8,9) (negative)")
print(f"  📊 Training Loss: {model.metrics['final_loss']:.4f}")
print(f"  📊 Training Accuracy: {model.metrics['final_accuracy']:.4f}")
print(f"  ⏱️  Training Time: {model.metrics['total_time']:.2f}s")

# Classification report
print(f"\n📝 Detailed Classification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# 6. Convergence Visualization
# -----------------------------
convergence_data = model.get_convergence_data()

# Plot 1: Loss convergence for sample classes
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
for i, cls in enumerate(list(convergence_data.keys())[:3]):
    plt.plot(convergence_data[cls]['loss_history'], label=f"Class {cls}", alpha=0.8)
plt.title("Loss Convergence - Sample Classes")
plt.xlabel("Iteration")
plt.ylabel("Hinge Loss")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Accuracy convergence for sample classes
plt.subplot(1, 3, 2)
for i, cls in enumerate(list(convergence_data.keys())[:3]):
    plt.plot(convergence_data[cls]['accuracy_history'], label=f"Class {cls}", alpha=0.8)
plt.title("Accuracy Convergence - Sample Classes")
plt.xlabel("Iteration")
plt.ylabel("Training Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Overall training progress
plt.subplot(1, 3, 3)
all_losses = []
for cls_data in convergence_data.values():
    all_losses.extend(cls_data['loss_history'])
plt.plot(all_losses, alpha=0.7, color='red')
plt.title("Overall Training Progress")
plt.xlabel("Total Iterations")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n✅ Scikit-learn SVM experiment completed!")
print(f"🏁 Final timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
