import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from transformers import AutoTokenizer, AutoModel
import numpy as np

class TextSVM(L.LightningModule):
    def __init__(self, model_name="distilbert-base-uncased", num_classes=2, lr_rate=0.01, C=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.lr_rate = lr_rate
        self.C = C  # regularization constant
        
        # Load pre-trained model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert_model = AutoModel.from_pretrained(model_name)
        
        # Freeze BERT parameters (optional - you can unfreeze for fine-tuning)
        for param in self.bert_model.parameters():
            param.requires_grad = False
            
        # Get BERT output dimension
        bert_dim = self.bert_model.config.hidden_size  # 768 for distilbert
        
        # Linear classifier on top of BERT embeddings
        self.classifier = nn.Linear(bert_dim, num_classes)

    def forward(self, texts):
        # Tokenize texts
        if isinstance(texts, list):
            # Handle list of strings
            inputs = self.tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors="pt"
            )
        else:
            # Handle single string
            inputs = self.tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors="pt"
            )
        
        # Move to same device as model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get BERT embeddings
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            # Use [CLS] token embedding (first token)
            embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # Classify
        logits = self.classifier(embeddings)
        return logits

    def hinge_loss(self, outputs, labels):
        # Convert labels to one-hot for binary classification
        y_onehot = F.one_hot(labels, num_classes=self.num_classes).float()
        
        # Correct class scores
        correct_class_scores = (outputs * y_onehot).sum(dim=1, keepdim=True)
        
        # Margin loss
        margins = torch.clamp(outputs - correct_class_scores + 1.0, min=0)
        margins = margins * (1 - y_onehot)  # ignore correct class
        loss = margins.sum(dim=1).mean()
        
        # Regularization
        reg = 0.5 * torch.norm(self.classifier.weight) ** 2
        return loss + self.C * reg

    def training_step(self, batch, batch_idx):
        texts, labels = batch["text"], batch["label"]
        outputs = self(texts)
        loss = self.hinge_loss(outputs, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        texts, labels = batch["text"], batch["label"]
        logits = self(texts)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log("test_acc", acc, prog_bar=True)
        return acc

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr_rate)
