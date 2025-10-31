import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L


class BinarySVMAligned(L.LightningModule):
    """Binary SVM aligned with SimpleTextSVM (for MNIST digit 0 vs non-zero)"""

    def __init__(self, input_size=28*28, out_channels=2, lr_rate=0.01, C=1.0):
        super().__init__()
        self.fc = nn.Linear(input_size, out_channels)   # ✅ Two outputs, like SimpleTextSVM
        self.lr_rate = lr_rate
        self.C = C

    def forward(self, x):
        if x.dim() == 3:
            batch_size, _, _ = x.size()
            x = x.view(batch_size, -1)
        else:
            x = x.view(x.size(0), -1)
        return self.fc(x)

    def convert_to_binary_labels(self, labels):
        """Convert MNIST digits to binary: 0 → 1 (positive), others → 0 (negative)"""
        return (labels == 0).long()

    def hinge_loss(self, outputs, labels):
        """Same hinge logic as SimpleTextSVM"""
        binary_labels = self.convert_to_binary_labels(labels)
        y_onehot = F.one_hot(binary_labels, num_classes=2).float()

        # Compute margin loss
        correct_scores = (outputs * y_onehot).sum(dim=1, keepdim=True)
        margins = torch.clamp(outputs - correct_scores + 1.0, min=0)
        margins = margins * (1 - y_onehot)
        loss = margins.sum(dim=1).mean()

        # Add regularization
        reg = 0.5 * torch.norm(self.fc.weight) ** 2
        return loss + self.C * reg

    def training_step(self, batch, batch_idx):
        x, y = batch["image"].float(), batch["label"]
        outputs = self(x)
        loss = self.hinge_loss(outputs, y)

        binary_labels = self.convert_to_binary_labels(y)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == binary_labels).float().mean()

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_binary_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch["image"].float(), batch["label"]
        logits = self(x)
        binary_labels = self.convert_to_binary_labels(y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == binary_labels).float().mean()

        # ✅ Log with both names for p2pfl logger compatibility
        self.log("test_acc", acc, prog_bar=True)
        self.log("binary_accuracy", acc, prog_bar=True)
        return acc

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr_rate)
