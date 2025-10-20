import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

class SimpleTextSVM(L.LightningModule):
    """Binary SVM for digit 0 vs non-zero classification"""
    
    def __init__(self, input_size=28*28, out_channels=2, lr_rate=0.01, C=1.0):
        super().__init__()
        self.fc = nn.Linear(input_size, out_channels)
        self.lr_rate = lr_rate
        self.C = C  # regularization constant

    def forward(self, x):
        # For now, just use the image data as-is
        # In a real implementation, you'd convert text to embeddings first
        batch_size, _, _ = x.size()
        x = x.view(batch_size, -1)
        return self.fc(x)

    def convert_to_binary_labels(self, labels):
        """Convert digit labels to binary: 0=1 (positive), others=0 (negative)"""
        return (labels == 0).long()
    
    def hinge_loss(self, outputs, labels):
        # Convert to binary labels: 0=1 (positive), others=0 (negative)
        binary_labels = self.convert_to_binary_labels(labels)
        
        # Convert to one-hot for binary classification
        y_onehot = F.one_hot(binary_labels, num_classes=2).float()
        
        # Correct class scores
        correct_class_scores = (outputs * y_onehot).sum(dim=1, keepdim=True)
        
        # Binary SVM margin loss
        margins = torch.clamp(outputs - correct_class_scores + 1.0, min=0)
        margins = margins * (1 - y_onehot)  # ignore correct class
        loss = margins.sum(dim=1).mean()
        
        # Regularization
        reg = 0.5 * torch.norm(self.fc.weight) ** 2
        return loss + self.C * reg

    def training_step(self, batch, batch_idx):
        x, y = batch["image"].float(), batch["label"]
        outputs = self(x)
        loss = self.hinge_loss(outputs, y)
        
        # Calculate binary accuracy for monitoring
        binary_labels = self.convert_to_binary_labels(y)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == binary_labels).float().mean()
        
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_binary_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch["image"].float(), batch["label"]
        logits = self(x)
        
        # Convert to binary labels for evaluation
        binary_labels = self.convert_to_binary_labels(y)
        preds = torch.argmax(logits, dim=1)
        
        # Calculate binary accuracy
        acc = (preds == binary_labels).float().mean()
        self.log("test_acc", acc, prog_bar=True)
        
        # Log additional metrics
        self.log("binary_accuracy", acc, prog_bar=True)
        return acc

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr_rate)
