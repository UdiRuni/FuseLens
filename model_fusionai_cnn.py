import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm.auto import tqdm

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
class Config:
    MAX_LEN = 32768
    BATCH_SIZE = 32      # CNNs are memory efficient; we can use a large batch
    EPOCHS = 3
    LR = 1e-4
    OUTPUT_DIR = "./checkpoints_fusionai_baseline"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# 2. ARCHITECTURE: FUSION-AI (1D CNN ResNet Surrogate)
# ==============================================================================
class GenomicResBlock(nn.Module):
    """A standard 1D Residual Block for genomic CNNs"""
    def __init__(self, channels, kernel_size=9):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return self.relu(x)

class FusionAIBaseline(nn.Module):
    """
    Replication of the FusionAI Deep CNN architecture for 32k window classification.
    Expects input shape: (Batch, Seq_Len) with integer tokens (0-4).
    """
    def __init__(self, vocab_size=5, embed_dim=4, num_filters=64):
        super().__init__()
        # 1. One-Hot style embedding (A, C, G, T, N)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 2. Large Receptive Field Convolution (as specified in FusionAI)
        self.stem = nn.Sequential(
            nn.Conv1d(embed_dim, num_filters, kernel_size=20, padding=10),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )
        
        # 3. Deep Neural Network Layers (Residual CNNs)
        self.res_blocks = nn.Sequential(
            GenomicResBlock(num_filters, kernel_size=9),
            nn.MaxPool1d(4),
            GenomicResBlock(num_filters, kernel_size=9),
            nn.MaxPool1d(4),
            GenomicResBlock(num_filters, kernel_size=9)
        )
        
        # 4. Global Pooling & Dense Classification Head
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(num_filters, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 2) # Binary output (Negative, Positive)

    def forward(self, x):
        # x shape: (Batch, Seq_Len)
        x = self.embedding(x)                 # -> (Batch, Seq_Len, Embed_Dim)
        x = x.transpose(1, 2)                 # -> (Batch, Embed_Dim, Seq_Len) for Conv1D
        
        x = self.stem(x)
        x = self.res_blocks(x)
        
        x = self.global_pool(x).squeeze(-1)   # -> (Batch, num_filters)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return {'logits': logits}

# ==============================================================================
# 3. SMART DATASET & MOCK GENERATOR
# ==============================================================================
def generate_mock_data(n=100):
    """Generates mock integer-encoded DNA sequences to verify pipeline"""
    data = []
    for _ in range(n):
        seq = torch.randint(0, 4, (Config.MAX_LEN,))
        label = random.choice([0, 1])
        data.append({'input_ids': seq, 'labels': torch.tensor(label, dtype=torch.long)})
    return data

class MockLoader:
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
    def __iter__(self):
        for i in range(0, len(self.data), self.batch_size):
            batch = self.data[i:i+self.batch_size]
            yield {
                'input_ids': torch.stack([x['input_ids'] for x in batch]),
                'labels': torch.stack([x['labels'] for x in batch])
            }
    def __len__(self):
        return len(self.data) // self.batch_size

# ==============================================================================
# 4. TRAINING LOOP
# ==============================================================================
def train_fusionai():
    print("🚀 Initializing FusionAI Baseline (1D-CNN)...")
    model = FusionAIBaseline().to(Config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR)
    criterion = nn.CrossEntropyLoss()
    
    # Replace these with your actual DataLoader from the FuseLens pipeline
    train_loader = MockLoader(generate_mock_data(200), Config.BATCH_SIZE)
    val_loader = MockLoader(generate_mock_data(50), Config.BATCH_SIZE)
    
    best_auc = 0.0
    
    for epoch in range(Config.EPOCHS):
        # --- TRAIN ---
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            ids = batch['input_ids'].to(Config.DEVICE)
            lbls = batch['labels'].to(Config.DEVICE)
            
            optimizer.zero_grad()
            out = model(ids)
            loss = criterion(out['logits'], lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # --- VALIDATE ---
        model.eval()
        all_preds, all_lbls = [], []
        with torch.no_grad():
            for batch in val_loader:
                ids = batch['input_ids'].to(Config.DEVICE)
                lbls = batch['labels'].to(Config.DEVICE)
                out = model(ids)
                
                probs = F.softmax(out['logits'], dim=1)[:, 1]
                all_preds.extend(probs.cpu().numpy())
                all_lbls.extend(lbls.cpu().numpy())
                
        auc = roc_auc_score(all_lbls, all_preds)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val AUC: {auc:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), os.path.join(Config.OUTPUT_DIR, "best_fusionai_cnn.pt"))

if __name__ == "__main__":
    train_fusionai()
