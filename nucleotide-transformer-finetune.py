# ==============================================================================
# CELL 1: IMPORTS & CONFIGURATION
# ==============================================================================
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW 
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, matthews_corrcoef, roc_auc_score, 
    average_precision_score, confusion_matrix, roc_curve, f1_score,
    precision_score, recall_score, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from datetime import datetime
from safetensors.torch import save_file
from collections import defaultdict

# --- CONFIGURATION ---
class Config:
    # Files (Update paths if necessary)
    POSITIVE_CSV = "datasets/fusion_gene_positive_bp_information_with_class_for_modeling.txt"
    NEGATIVE_CSV = "datasets/fusion_gene_negative_bp_information_with_class_for_modeling.txt"
    TEST_CSV = "datasets/fusion_gene_positive_bp_information_with_class_for_testing.txt"
    
    # Extra FASTA Data
    EXTRA_POS_FASTA = "datasets/blast_validated_chimeras.fasta,datasets/cosmic_high_confidence_sequences.fna,datasets/chimeras_43466.fa"
    NEG_FASTA_CANONICAL = "datasets/false_negative_candidates.fasta"
    NEG_FASTA_SYNTHETIC = "datasets/false_positive_candidates.fasta"
    
    # Model & Training
    OUTPUT_DIR = "./nucleotide_transformer_v1_checkpoints_sliding"
    
    # --- NUCLEOTIDE TRANSFORMER MODEL ---
    MODEL_NAME = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
    
    # --- SLIDING WINDOW CONFIGURATION ---
    WINDOW_BP = 12000       # Window size in base pairs
    STRIDE_BP = 6000        # Stride for sliding window (50% overlap)
    MAX_TOKENS = 2048       # Max tokens per window (WINDOW_BP / 6 + buffer)
    
    # --- ORIGINAL SEQUENCE LENGTH ---
    ORIGINAL_SEQ_LEN = 32000  # Your original 32k sequences
    
    BATCH_SIZE = 4
    EPOCHS = 3
    LEARNING_RATE = 1e-5
    VAL_SPLIT = 0.2
    SEED = 42
    
    CONFIDENCE_THRESHOLD = 0.90
    
    # Aggregation strategy: 'max', 'mean', or 'attention_weighted'
    AGGREGATION_STRATEGY = 'max'

if not os.path.exists(Config.OUTPUT_DIR):
    os.makedirs(Config.OUTPUT_DIR)

# --- REPRODUCIBILITY ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(Config.SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Setup Complete. Device: {device}")

if torch.cuda.is_available():
    print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

print(f"\n📐 Sliding Window Config:")
print(f"   Window Size: {Config.WINDOW_BP} bp")
print(f"   Stride: {Config.STRIDE_BP} bp ({Config.STRIDE_BP/Config.WINDOW_BP*100:.0f}% overlap)")
print(f"   Original Seq Length: {Config.ORIGINAL_SEQ_LEN} bp")
n_windows = (Config.ORIGINAL_SEQ_LEN - Config.WINDOW_BP) // Config.STRIDE_BP + 1
print(f"   Expected Windows per Sequence: ~{n_windows}")


# ==============================================================================
# CELL 2: CLASS DEFINITIONS (DataPrep, SlidingWindowDataset, Model)
# ==============================================================================

# --- 1. DATA PREPARATOR ---
class DataPreparator:
    COLUMNS = ["Hgene","Hchr","Hbp","Hstrand","Tgene","Tchr","Tbp","Tstrand","5'-gene sequence (10Kb)","3'-gene sequence (10Kb)"]

    def __init__(self, config):
        self.cfg = config
        self.trans_table = str.maketrans("ATCGN", "TAGCN")

    def _trim_artifacts(self, sequence: str) -> str:
        if not isinstance(sequence, str): return ""
        return sequence.upper().strip()

    def _get_reverse_complement(self, sequence: str) -> str:
        return sequence.upper().translate(self.trans_table)[::-1]

    def _generate_random_dna(self, length: int) -> str:
        if length <= 0: return ""
        return "".join(random.choices("ACGT", k=length))

    def _prepare_full_sequence(self, seq_5p: str, seq_3p: str) -> str:
        """
        Combines 5' and 3' sequences into full sequence.
        Pads to ORIGINAL_SEQ_LEN if shorter, crops if longer.
        NO windowing here - that's handled by the Dataset.
        """
        seq_5p = self._trim_artifacts(seq_5p)
        seq_3p = self._trim_artifacts(seq_3p)
        
        core_seq = seq_5p + seq_3p
        core_len = len(core_seq)
        target_len = self.cfg.ORIGINAL_SEQ_LEN
        
        if core_len < target_len:
            # Pad with random DNA on both sides
            needed = target_len - core_len
            pad_left = random.randint(0, needed)
            pad_right = needed - pad_left
            left_seq = self._generate_random_dna(pad_left)
            right_seq = self._generate_random_dna(pad_right)
            return left_seq + core_seq + right_seq
        elif core_len > target_len:
            # Random crop for very long sequences
            max_start = core_len - target_len
            start = random.randint(0, max_start)
            return core_seq[start : start + target_len]
        else:
            return core_seq

    def _load_fasta(self, path: str) -> list:
        if not os.path.exists(path): return []
        seqs, curr = [], []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if curr: seqs.append(self._trim_artifacts("".join(curr)))
                    curr = []
                else: curr.append(line)
            if curr: seqs.append(self._trim_artifacts("".join(curr)))
        return seqs


# --- 2. SLIDING WINDOW DATASET ---
class SlidingWindowDataset(Dataset):
    """
    Creates overlapping windows from long sequences for Nucleotide Transformer.
    Each window is treated as a separate sample during training.
    Tracks metadata for aggregation during inference.
    """
    
    def __init__(self, sequences, labels, tokenizer, window_bp, stride_bp, max_tokens, 
                 is_inference=False):
        """
        Args:
            sequences: List of full-length DNA sequences
            labels: List of labels (0 or 1)
            tokenizer: HuggingFace tokenizer
            window_bp: Window size in base pairs
            stride_bp: Stride between windows in base pairs
            max_tokens: Maximum tokens for tokenizer
            is_inference: If True, creates all windows. If False, randomly samples one window per epoch.
        """
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.window_bp = window_bp
        self.stride_bp = stride_bp
        self.max_tokens = max_tokens
        self.is_inference = is_inference
        
        # Pre-compute window information
        self.windows = []  # List of (seq_idx, window_start, window_end)
        self.window_to_seq = []  # Maps window index to original sequence index
        
        if is_inference:
            # For inference: create ALL windows for complete coverage
            self._create_all_windows()
        else:
            # For training: we'll sample windows dynamically
            self._create_window_indices()
    
    def _create_all_windows(self):
        """Create all overlapping windows for each sequence (inference mode)."""
        for seq_idx, seq in enumerate(self.sequences):
            seq_len = len(seq)
            
            if seq_len <= self.window_bp:
                # Single window for short sequences
                self.windows.append((seq_idx, 0, seq_len))
                self.window_to_seq.append(seq_idx)
            else:
                # Sliding windows
                for start in range(0, seq_len - self.window_bp + 1, self.stride_bp):
                    end = start + self.window_bp
                    self.windows.append((seq_idx, start, end))
                    self.window_to_seq.append(seq_idx)
                
                # Ensure we capture the very end
                last_start = seq_len - self.window_bp
                if last_start % self.stride_bp != 0:
                    self.windows.append((seq_idx, last_start, seq_len))
                    self.window_to_seq.append(seq_idx)
        
        print(f"   Created {len(self.windows)} windows from {len(self.sequences)} sequences")
        print(f"   Average windows per sequence: {len(self.windows)/len(self.sequences):.1f}")
    
    def _create_window_indices(self):
        """For training: just track sequence indices, sample windows dynamically."""
        self.seq_indices = list(range(len(self.sequences)))
    
    def _extract_window(self, seq, seq_len):
        """Extract a random window from sequence (training mode)."""
        if seq_len <= self.window_bp:
            # Pad short sequences
            needed = self.window_bp - seq_len
            pad_left = random.randint(0, needed)
            pad_right = needed - pad_left
            padded = 'N' * pad_left + seq + 'N' * pad_right
            return padded, pad_left  # Return pad_left as window_start offset
        else:
            # Random window selection
            max_start = seq_len - self.window_bp
            start = random.randint(0, max_start)
            return seq[start:start + self.window_bp], start
    
    def __len__(self):
        if self.is_inference:
            return len(self.windows)
        else:
            return len(self.sequences)
    
    def __getitem__(self, idx):
        if self.is_inference:
            # Inference mode: use pre-computed windows
            seq_idx, win_start, win_end = self.windows[idx]
            seq = str(self.sequences[seq_idx])
            label = self.labels[seq_idx]
            
            window_seq = seq[win_start:win_end]
            
            # Pad if needed (for edge windows)
            if len(window_seq) < self.window_bp:
                needed = self.window_bp - len(window_seq)
                window_seq = window_seq + 'N' * needed
        else:
            # Training mode: random window sampling
            seq_idx = idx
            seq = str(self.sequences[seq_idx])
            label = self.labels[seq_idx]
            seq_len = len(seq)
            
            window_seq, win_start = self._extract_window(seq, seq_len)
        
        # Tokenize
        enc = self.tokenizer(
            window_seq,
            truncation=True,
            max_length=self.max_tokens,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = enc['input_ids'].squeeze(0)
        
        if 'attention_mask' in enc:
            mask = enc['attention_mask'].squeeze(0)
        else:
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            mask = (input_ids != pad_id).long()
        
        result = {
            'input_ids': input_ids,
            'attention_mask': mask,
            'labels': torch.tensor(label, dtype=torch.long),
            'seq_idx': torch.tensor(seq_idx, dtype=torch.long),
            'window_start': torch.tensor(win_start if self.is_inference else 0, dtype=torch.long)
        }
        
        return result


# --- 3. NUCLEOTIDE TRANSFORMER CLASSIFIER WITH ATTENTION POOLING ---
class NucleotideTransformerClassifier(nn.Module):
    def __init__(self, model_name, num_labels=2):
        super().__init__()
        
        # Load Nucleotide Transformer base model
        self.nt_model = AutoModelForMaskedLM.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        
        # Get hidden size from config
        if hasattr(self.nt_model.config, 'hidden_size'):
            hidden_size = self.nt_model.config.hidden_size
        elif hasattr(self.nt_model.config, 'd_model'):
            hidden_size = self.nt_model.config.d_model
        else:
            hidden_size = 1024
            
        print(f"   Model hidden size: {hidden_size}")
        
        # Attention Pooling Head
        self.attention_head = nn.Linear(hidden_size, 1)
        
        # Classification Head
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        self.hidden_size = hidden_size
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        # 1. Get embeddings from NT model
        outputs = self.nt_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Get last hidden state
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[-1]
        else:
            hidden_states = self.nt_model.esm.encoder(
                input_ids, 
                attention_mask=attention_mask
            ).last_hidden_state
        
        # 2. Attention Pooling Mechanism
        attention_scores = self.attention_head(hidden_states)
        
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(
                attention_mask.unsqueeze(-1) == 0, 
                float('-inf')
            )
        
        attention_probs = torch.softmax(attention_scores, dim=1)
        
        # 3. Weighted pooling
        pooled_output = torch.sum(hidden_states * attention_probs, dim=1)
        
        # 4. Classification
        logits = self.classifier(pooled_output)
        
        # 5. Loss calculation
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        # 6. Breakpoint estimation (max attention index within window)
        bp_index = torch.argmax(attention_probs, dim=1).squeeze(-1)
        
        # 7. Confidence score (max attention value - indicates how focused the model is)
        max_attention = torch.max(attention_probs, dim=1).values.squeeze(-1)
        
        return {
            'loss': loss,
            'logits': logits,
            'breakpoint_index': bp_index,
            'attention_probs': attention_probs.squeeze(-1),
            'max_attention': max_attention
        }


# ==============================================================================
# CELL 3: WINDOW AGGREGATION LOGIC
# ==============================================================================

class WindowAggregator:
    """
    Aggregates predictions from multiple windows back to sequence-level predictions.
    """
    
    def __init__(self, strategy='max'):
        """
        Args:
            strategy: 'max' - take max probability across windows
                     'mean' - average probabilities
                     'attention_weighted' - weight by model's attention confidence
        """
        self.strategy = strategy
    
    def aggregate(self, seq_indices, window_starts, probs, bp_indices, 
                  attention_probs=None, max_attentions=None, labels=None):
        """
        Aggregate window-level predictions to sequence-level.
        
        Returns:
            seq_probs: (n_sequences, 2) aggregated probabilities
            seq_labels: (n_sequences,) labels
            seq_bp_global: (n_sequences,) global breakpoint positions in bp
            seq_bp_confidence: (n_sequences,) confidence of breakpoint prediction
        """
        # Group by sequence index
        seq_data = defaultdict(list)
        
        for i in range(len(seq_indices)):
            seq_idx = seq_indices[i]
            seq_data[seq_idx].append({
                'prob': probs[i],
                'window_start': window_starts[i],
                'bp_index': bp_indices[i],
                'max_attention': max_attentions[i] if max_attentions is not None else 1.0,
                'label': labels[i] if labels is not None else None
            })
        
        # Aggregate
        n_seqs = max(seq_data.keys()) + 1
        seq_probs = np.zeros((n_seqs, 2))
        seq_labels = np.zeros(n_seqs, dtype=int)
        seq_bp_global = np.zeros(n_seqs, dtype=int)
        seq_bp_confidence = np.zeros(n_seqs)
        
        for seq_idx, windows in seq_data.items():
            # Stack probabilities
            window_probs = np.array([w['prob'] for w in windows])
            
            if self.strategy == 'max':
                # Take the window with highest fusion probability
                best_window_idx = np.argmax(window_probs[:, 1])
                seq_probs[seq_idx] = window_probs[best_window_idx]
                
            elif self.strategy == 'mean':
                # Average across all windows
                seq_probs[seq_idx] = np.mean(window_probs, axis=0)
                best_window_idx = np.argmax(window_probs[:, 1])
                
            elif self.strategy == 'attention_weighted':
                # Weight by attention confidence
                weights = np.array([w['max_attention'] for w in windows])
                weights = weights / weights.sum()
                seq_probs[seq_idx] = np.average(window_probs, axis=0, weights=weights)
                best_window_idx = np.argmax(weights)
            
            # Get label (same for all windows of a sequence)
            if windows[0]['label'] is not None:
                seq_labels[seq_idx] = windows[0]['label']
            
            # Calculate global breakpoint position
            best_window = windows[best_window_idx]
            # Convert token index to bp: token_idx * 6 (approximate for 6-mer)
            local_bp = best_window['bp_index'] * 6
            global_bp = best_window['window_start'] + local_bp
            seq_bp_global[seq_idx] = global_bp
            seq_bp_confidence[seq_idx] = best_window['max_attention']
        
        return seq_probs, seq_labels, seq_bp_global, seq_bp_confidence


# ==============================================================================
# CELL 4: METRICS & VISUALIZATION LOGIC
# ==============================================================================

def compute_metrics(y_true, y_probs):
    y_pred = np.argmax(y_probs, axis=1)
    
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    
    try: 
        auroc = roc_auc_score(y_true, y_probs[:, 1])
    except ValueError: 
        auroc = 0.0
        
    return {
        "acc": acc, "mcc": mcc, "f1": f1, "prec": prec, "rec": rec, "auroc": auroc
    }

def plot_comprehensive_results(y_true, y_probs, predicted_bps, title_prefix=""):
    y_pred = np.argmax(y_probs, axis=1)
    y_scores = y_probs[:, 1]
    
    plt.figure(figsize=(14, 10)) 
    
    # A. Confusion Matrix
    plt.subplot(2, 2, 1)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
    plt.title(f'{title_prefix} Confusion Matrix')

    # B. ROC Curve
    plt.subplot(2, 2, 2)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_true, y_scores):.4f}', color='purple', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.legend()
    plt.title(f'{title_prefix} ROC Curve')

    # C. Probability Distribution
    plt.subplot(2, 2, 3)
    neg_scores = y_scores[y_true == 0]
    pos_scores = y_scores[y_true == 1]
    sns.histplot(neg_scores, color='red', alpha=0.5, label='True Neg', bins=30, kde=True)
    sns.histplot(pos_scores, color='green', alpha=0.5, label='True Pos', bins=30, kde=True)
    plt.axvline(0.5, color='black', linestyle='--')
    plt.legend()
    plt.title(f'{title_prefix} Probability Dist.')

    # D. Breakpoint Locations (Global BP coordinates)
    plt.subplot(2, 2, 4)
    high_conf = np.where((y_pred == 1) & (y_scores > Config.CONFIDENCE_THRESHOLD))[0]
    if len(high_conf) > 0:
        sns.histplot(predicted_bps[high_conf], kde=True, bins=50, color='blue', label='Predicted')
        plt.title(f'{title_prefix} Breakpoints (Conf > {Config.CONFIDENCE_THRESHOLD})')
        plt.xlabel('Global Position (bp)')
    else:
        plt.text(0.5, 0.5, "No High-Confidence Fusions", ha='center')
        plt.title('Breakpoint Locations')

    plt.tight_layout()
    plt.show()


# ==============================================================================
# CELL 5: TRAINING & VALIDATION LOOPS (WITH SLIDING WINDOW)
# ==============================================================================

def train_epoch(model, loader, optimizer, scheduler, device):
    """
    Training loop - each window is an independent sample.
    No aggregation needed during training.
    """
    model.train()
    total_loss = 0
    all_probs, all_labels = [], []
    
    for batch in tqdm(loader, desc="Training"):
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        lbls = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(ids, attention_mask=mask, labels=lbls)
        loss = outputs['loss']
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        probs = torch.softmax(outputs['logits'], dim=1)
        
        all_probs.extend(probs.detach().cpu().numpy())
        all_labels.extend(lbls.detach().cpu().numpy())
        
        del ids, mask, lbls, outputs, loss
    
    if torch.cuda.is_available(): 
        torch.cuda.empty_cache()
    
    # Compute window-level metrics (for monitoring)
    metrics = compute_metrics(np.array(all_labels), np.array(all_probs))
    metrics['loss'] = total_loss / len(loader)
    return metrics


def validate_epoch_with_aggregation(model, loader, criterion, device, aggregator, desc="Validating"):
    """
    Validation/Test loop with window aggregation.
    Collects all window predictions, then aggregates to sequence level.
    """
    model.eval()
    
    # Collect window-level data
    all_seq_indices = []
    all_window_starts = []
    all_probs = []
    all_labels = []
    all_bp_indices = []
    all_max_attentions = []
    all_attention_maps = []
    
    val_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            lbls = batch['labels'].to(device)
            seq_idx = batch['seq_idx']
            win_start = batch['window_start']
            
            outputs = model(ids, attention_mask=mask, labels=lbls)
            val_loss += outputs['loss'].item()
            
            probs = torch.softmax(outputs['logits'], dim=1)
            bps = outputs['breakpoint_index']
            max_attn = outputs['max_attention']
            attn_maps = outputs['attention_probs']
            
            # Collect
            all_seq_indices.extend(seq_idx.cpu().numpy())
            all_window_starts.extend(win_start.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(lbls.cpu().numpy())
            all_bp_indices.extend(bps.cpu().numpy())
            all_max_attentions.extend(max_attn.cpu().numpy())
            all_attention_maps.extend(attn_maps.cpu().numpy())
            
            del ids, mask, lbls, outputs
    
    # Convert to arrays
    all_seq_indices = np.array(all_seq_indices)
    all_window_starts = np.array(all_window_starts)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_bp_indices = np.array(all_bp_indices)
    all_max_attentions = np.array(all_max_attentions)
    
    # Aggregate to sequence level
    seq_probs, seq_labels, seq_bp_global, seq_bp_confidence = aggregator.aggregate(
        all_seq_indices, all_window_starts, all_probs, all_bp_indices,
        max_attentions=all_max_attentions, labels=all_labels
    )
    
    # Compute sequence-level metrics
    metrics = compute_metrics(seq_labels, seq_probs)
    metrics['loss'] = val_loss / len(loader)
    
    # Also compute FWHM-style width stats (approximate from attention)
    # We'll use the best window's attention for width calculation
    seq_widths = []
    seq_starts = []
    seq_ends = []
    
    for seq_idx in range(len(seq_probs)):
        # Find windows for this sequence
        window_mask = all_seq_indices == seq_idx
        if not np.any(window_mask):
            seq_widths.append(0)
            seq_starts.append(0)
            seq_ends.append(0)
            continue
            
        # Get the best window (highest fusion prob)
        window_indices = np.where(window_mask)[0]
        best_local_idx = np.argmax(all_probs[window_mask, 1])
        best_window_global_idx = window_indices[best_local_idx]
        
        attn_map = all_attention_maps[best_window_global_idx]
        peak_idx = all_bp_indices[best_window_global_idx]
        peak_val = attn_map[peak_idx] if peak_idx < len(attn_map) else attn_map.max()
        
        # FWHM calculation
        threshold = peak_val * 0.5
        above_thresh = np.where(attn_map >= threshold)[0]
        
        if len(above_thresh) > 0:
            start_tok = above_thresh.min()
            end_tok = above_thresh.max()
            width_tok = end_tok - start_tok
        else:
            start_tok, end_tok, width_tok = peak_idx, peak_idx, 0
        
        # Convert to global bp coordinates
        win_start = all_window_starts[best_window_global_idx]
        global_start = win_start + start_tok * 6
        global_end = win_start + end_tok * 6
        global_width = width_tok * 6
        
        seq_starts.append(global_start)
        seq_ends.append(global_end)
        seq_widths.append(global_width)
    
    return (metrics, seq_labels, seq_probs, seq_bp_global, 
            (np.array(seq_starts), np.array(seq_ends), np.array(seq_widths)),
            seq_bp_confidence)


# ==============================================================================
# CELL 6: LOAD DATA & PERFORM STRICT GENE-PAIR SPLIT
# ==============================================================================
import hashlib

print("\n--- 1. LOADING ALL DATA ---")
preparator = DataPreparator(Config)

all_pos_lengths = []
all_neg_lengths = []

# =======================================================
# A. LOAD POSITIVES (CSV + FASTA)
# =======================================================
pos_dfs = []

if os.path.exists(Config.POSITIVE_CSV):
    try:
        print(f"  > Loading Positive CSV: {Config.POSITIVE_CSV}")
        csv_df = pd.read_csv(Config.POSITIVE_CSV, header=None, names=DataPreparator.COLUMNS, sep='\t')
        
        raw_lens = (
            csv_df["5'-gene sequence (10Kb)"].astype(str).str.strip().str.len() + 
            csv_df["3'-gene sequence (10Kb)"].astype(str).str.strip().str.len()
        )
        all_pos_lengths.extend(raw_lens.tolist())

        # Prepare FULL sequences (no cropping yet)
        csv_df['sequence'] = csv_df.apply(
            lambda r: preparator._prepare_full_sequence(r["5'-gene sequence (10Kb)"], r["3'-gene sequence (10Kb)"]), 
            axis=1
        )
        csv_df['label'] = 1
        csv_df['fusion_name'] = csv_df['Hgene'] + "--" + csv_df['Tgene']
        
        pos_dfs.append(csv_df)
        print(f"    - Loaded {len(csv_df)} samples from CSV.")
    except Exception as e:
        print(f"    ⚠️ Error reading Positive CSV: {e}")

if hasattr(Config, 'EXTRA_POS_FASTA') and Config.EXTRA_POS_FASTA:
    print("  > Loading Extra FASTA Positives...")
    extra_seqs = []
    for p in Config.EXTRA_POS_FASTA.split(','):
        p = p.strip()
        if p and os.path.exists(p):
            loaded_raw = preparator._load_fasta(p)
            all_pos_lengths.extend([len(s) for s in loaded_raw])
            processed = [preparator._prepare_full_sequence(s, "") for s in loaded_raw]
            extra_seqs.extend(processed)
            print(f"    - Loaded {len(processed)} from {os.path.basename(p)}")
            
    if extra_seqs:
        pos_dfs.append(pd.DataFrame({'sequence': extra_seqs, 'label': 1}))

if pos_dfs:
    pos_df = pd.concat(pos_dfs, ignore_index=True)
    pos_df.drop_duplicates(subset=['sequence'], inplace=True)
else:
    pos_df = pd.DataFrame(columns=['sequence', 'label'])

# =======================================================
# B. LOAD NEGATIVES (CSV + FASTA)
# =======================================================
neg_dfs = []

if os.path.exists(Config.NEGATIVE_CSV):
    try:
        print(f"  > Loading Negative CSV: {Config.NEGATIVE_CSV}")
        neg_csv = pd.read_csv(Config.NEGATIVE_CSV, header=None, names=DataPreparator.COLUMNS, sep='\t')
        
        raw_lens_neg = (
            neg_csv["5'-gene sequence (10Kb)"].astype(str).str.strip().str.len() + 
            neg_csv["3'-gene sequence (10Kb)"].astype(str).str.strip().str.len()
        )
        all_neg_lengths.extend(raw_lens_neg.tolist())

        neg_csv['sequence'] = neg_csv.apply(
            lambda r: preparator._prepare_full_sequence(r["5'-gene sequence (10Kb)"], r["3'-gene sequence (10Kb)"]), 
            axis=1
        )
        neg_csv['label'] = 0
        neg_csv['gene_name'] = neg_csv['Hgene']
        
        print("    > Augmenting CSV Negatives (Reverse Complement)...")
        aug_csv = neg_csv.copy()
        aug_csv['sequence'] = aug_csv['sequence'].apply(preparator._get_reverse_complement)
        neg_csv = pd.concat([neg_csv, aug_csv], ignore_index=True)
        
        neg_dfs.append(neg_csv)
    except Exception as e:
        print(f"    ⚠️ Error reading Negative CSV: {e}")

neg_fasta_sources = [Config.NEG_FASTA_CANONICAL, Config.NEG_FASTA_SYNTHETIC]
for p in neg_fasta_sources:
    if os.path.exists(p):
        print(f"  > Loading Negative FASTA: {os.path.basename(p)}")
        raw_negs = preparator._load_fasta(p)
        all_neg_lengths.extend([len(s) for s in raw_negs])
        processed_negs = [preparator._prepare_full_sequence(s, "") for s in raw_negs]
        neg_dfs.append(pd.DataFrame({'sequence': processed_negs, 'label': 0}))

if neg_dfs:
    neg_df = pd.concat(neg_dfs, ignore_index=True)
    neg_df.drop_duplicates(subset=['sequence'], inplace=True)
else:
    neg_df = pd.DataFrame(columns=['sequence', 'label'])

# =======================================================
# C. STRICT GENE-PAIR HOLDOUT SPLIT
# =======================================================
print("\n--- PERFORMING STRICT BIOLOGICAL SPLIT ---")
full_df = pd.concat([pos_df, neg_df], ignore_index=True)

def get_biological_group_id(row):
    if row['label'] == 1 and 'fusion_name' in row and pd.notna(row.get('fusion_name')):
        return f"POS_{row['fusion_name']}"
    elif row['label'] == 0 and 'gene_name' in row and pd.notna(row.get('gene_name')):
        return f"NEG_{row['gene_name']}"
    s = str(row['sequence']).upper().strip()
    return hashlib.md5(s.encode()).hexdigest()

full_df['split_group_id'] = full_df.apply(get_biological_group_id, axis=1)
unique_groups = full_df['split_group_id'].unique()

print(f"✅ Total Unique Samples: {len(full_df)}")
print(f"🧬 Unique Biological Groups: {len(unique_groups)}")

train_groups, temp_groups = train_test_split(unique_groups, test_size=0.30, random_state=Config.SEED)
val_groups, test_groups = train_test_split(temp_groups, test_size=0.50, random_state=Config.SEED)

train_df = full_df[full_df['split_group_id'].isin(train_groups)].copy()
val_df = full_df[full_df['split_group_id'].isin(val_groups)].copy()
test_df = full_df[full_df['split_group_id'].isin(test_groups)].copy()

print(f"\n--- FINAL SPLIT SIZES ---")
print(f"TRAIN: {len(train_df)} sequences")
print(f"VAL:   {len(val_df)} sequences")
print(f"TEST:  {len(test_df)} sequences")

overlap = set(train_df['split_group_id']).intersection(set(test_df['split_group_id']))
if not overlap:
    print("✅ LEAKAGE CHECK PASSED: No biological events shared between Train and Test.")
else:
    print(f"❌ LEAKAGE DETECTED: {len(overlap)} shared groups!")


# ==============================================================================
# CELL 7: INITIALIZE & TRAIN MODEL
# ==============================================================================
from safetensors.torch import save_file 

train_seqs_list = train_df['sequence'].tolist()
train_lbls_list = train_df['label'].tolist()
val_seqs_list = val_df['sequence'].tolist()
val_lbls_list = val_df['label'].tolist()

n_train_pos = sum(train_lbls_list)
n_train_neg = len(train_lbls_list) - n_train_pos
n_val_pos = sum(val_lbls_list)
n_val_neg = len(val_lbls_list) - n_val_pos

print(f"--- DATASET STATISTICS ---")
print(f"Training Set:   {len(train_seqs_list)} sequences")
print(f"  ├── Positives: {n_train_pos} ({n_train_pos/len(train_seqs_list):.1%})")
print(f"  └── Negatives: {n_train_neg} ({n_train_neg/len(train_seqs_list):.1%})")
print(f"Validation Set: {len(val_seqs_list)} sequences")
print(f"  ├── Positives: {n_val_pos} ({n_val_pos/len(val_seqs_list):.1%})")
print(f"  └── Negatives: {n_val_neg} ({n_val_neg/len(val_seqs_list):.1%})")

# Initialize Tokenizer & Model
print(f"\n🔬 Initializing Nucleotide Transformer: {Config.MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else '[PAD]'

model = NucleotideTransformerClassifier(Config.MODEL_NAME)
model.to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   Total Parameters: {total_params:,}")
print(f"   Trainable Parameters: {trainable_params:,}")

# Create Datasets with Sliding Window
print(f"\n📐 Creating Sliding Window Datasets...")

# Training: Random window sampling (is_inference=False)
train_dataset = SlidingWindowDataset(
    train_seqs_list, train_lbls_list, tokenizer,
    window_bp=Config.WINDOW_BP, stride_bp=Config.STRIDE_BP, 
    max_tokens=Config.MAX_TOKENS, is_inference=False
)

# Validation & Test: All windows for aggregation (is_inference=True)
val_dataset = SlidingWindowDataset(
    val_seqs_list, val_lbls_list, tokenizer,
    window_bp=Config.WINDOW_BP, stride_bp=Config.STRIDE_BP,
    max_tokens=Config.MAX_TOKENS, is_inference=True
)

train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

print(f"   Train: {len(train_dataset)} samples (1 random window/seq/epoch)")
print(f"   Val: {len(val_dataset)} windows (all windows for aggregation)")

# Create Test Loader
if 'test_df' in globals():
    test_seqs_list = test_df['sequence'].tolist()
    test_lbls_list = test_df['label'].tolist()
    test_dataset = SlidingWindowDataset(
        test_seqs_list, test_lbls_list, tokenizer,
        window_bp=Config.WINDOW_BP, stride_bp=Config.STRIDE_BP,
        max_tokens=Config.MAX_TOKENS, is_inference=True
    )
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    print(f"   Test: {len(test_dataset)} windows")

# Initialize Aggregator
aggregator = WindowAggregator(strategy=Config.AGGREGATION_STRATEGY)
print(f"\n🔗 Aggregation Strategy: {Config.AGGREGATION_STRATEGY}")

# Optimizer & Scheduler
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
total_steps = len(train_loader) * Config.EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)

# Training Loop
print(f"\n🚀 Training Start ({Config.EPOCHS} Epochs)...")
best_val_auc = 0.0

for epoch in range(Config.EPOCHS):
    print(f"\n--- Epoch {epoch + 1} ---")
    
    # Train (window-level)
    train_metrics = train_epoch(model, train_loader, optimizer, scheduler, device)
    print(f"TRAIN (window) >> Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['acc']:.4f} | F1: {train_metrics['f1']:.4f}")
    
    # Validate (sequence-level with aggregation)
    val_results = validate_epoch_with_aggregation(model, val_loader, criterion, device, aggregator)
    val_metrics, val_true, val_probs, val_bps, (val_starts, val_ends, val_widths), val_conf = val_results
    print(f"VAL (seq-agg)  >> Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['acc']:.4f} | F1: {val_metrics['f1']:.4f} | AUC: {val_metrics['auroc']:.4f}")
    
    if val_metrics['auroc'] > best_val_auc:
        best_val_auc = val_metrics['auroc']
        save_path = os.path.join(Config.OUTPUT_DIR, "best_model.safetensors")
        state_dict_to_save = {k: v.clone() for k, v in model.state_dict().items()}
        save_file(state_dict_to_save, save_path)
        print(f"  ★ New Best Model Saved (AUC: {best_val_auc:.4f})")


# ==============================================================================
# CELL 8: BLIND TEST SET EVALUATION
# ==============================================================================
from safetensors.torch import load_file

print(f"\n{'='*50}")
print(f"PHASE: BLIND TEST SET EVALUATION (SLIDING WINDOW)")
print(f"{'='*50}")

if 'model' not in locals():
    print("🔄 Initializing Model Architecture...")
    model = NucleotideTransformerClassifier(Config.MODEL_NAME)

model.to(device)

best_model_path = os.path.join(Config.OUTPUT_DIR, "best_model.safetensors")

if not os.path.exists(best_model_path):
    pt_path = os.path.join(Config.OUTPUT_DIR, "best_model.pt")
    if os.path.exists(pt_path):
        best_model_path = pt_path

if os.path.exists(best_model_path):
    print(f"📂 Loading Best Model: {best_model_path}")
    try:
        if best_model_path.endswith('.safetensors'):
            state_dict = load_file(best_model_path)
        else:
            state_dict = torch.load(best_model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("✅ Weights loaded successfully.")
    except Exception as e:
        print(f"⚠️ Warning: Failed to load weights ({e}).")
else:
    print("⚠️ Warning: Best model file not found.")

model.eval()

if 'criterion' not in locals(): 
    criterion = torch.nn.CrossEntropyLoss()

if 'aggregator' not in locals():
    aggregator = WindowAggregator(strategy=Config.AGGREGATION_STRATEGY)

if 'test_loader' in locals():
    test_results = validate_epoch_with_aggregation(
        model, test_loader, criterion, device, aggregator, desc="Testing"
    )
    test_metrics, t_true, t_probs, t_bps, (t_starts, t_ends, t_widths), t_conf = test_results

    print(f"\n🏆 FINAL TEST RESULTS (Sequence-Level, Aggregated):")
    print(f"   Accuracy:  {test_metrics['acc']:.4f}")
    print(f"   AUC:       {test_metrics['auroc']:.4f}")
    print(f"   Precision: {test_metrics['prec']:.4f}")
    print(f"   Recall:    {test_metrics['rec']:.4f}")
    print(f"   F1 Score:  {test_metrics['f1']:.4f}")
    print(f"   MCC:       {test_metrics['mcc']:.4f}")

    # Export Results
    print(f"\n💾 EXPORTING DETAILED RESULTS...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"blind_test_predictions_sliding_{timestamp}.csv"

    # Create Results DataFrame
    if 'test_df' in locals() and len(test_df) == len(t_true):
        results_df = test_df.copy().reset_index(drop=True)
    else:
        results_df = pd.DataFrame()

    pred_indices = np.argmax(t_probs, axis=1)
    results_df['True_Label'] = t_true
    results_df['Predicted_Label'] = pred_indices
    results_df['Fusion_Prob'] = t_probs[:, 1]
    results_df['BP_Peak_Global'] = t_bps  # Global BP position
    results_df['BP_Region_Start'] = t_starts.astype(int)
    results_df['BP_Region_End'] = t_ends.astype(int)
    results_df['BP_Region_Width'] = t_widths.astype(int)
    results_df['BP_Confidence'] = t_conf

    results_df['Result_Type'] = results_df.apply(
        lambda x: 'TP' if (x['True_Label'] == 1 and x['Predicted_Label'] == 1) else
                  ('TN' if (x['True_Label'] == 0 and x['Predicted_Label'] == 0) else
                  ('FP' if (x['True_Label'] == 0 and x['Predicted_Label'] == 1) else 'FN')),
        axis=1
    )

    save_path = os.path.join(Config.OUTPUT_DIR, csv_filename)
    generic_path = os.path.join(Config.OUTPUT_DIR, "final_test_results_detailed.csv")
    
    results_df.to_csv(save_path, index=False)
    results_df.to_csv(generic_path, index=False)
    
    print(f"✅ Predictions saved to: {save_path}")

    # Visualization
    plot_comprehensive_results(t_true, t_probs, t_bps, title_prefix="Test (Sliding Window)")
    
    # Alias for downstream cells
    test_true = t_true
    test_probs = t_probs
    print("\n✅ Data variables ready for further analysis.")

else:
    print("❌ Critical Error: 'test_loader' not defined.")


# ==============================================================================
# CELL 9: DETAILED ERROR & GEOMETRIC ANALYSIS
# ==============================================================================
filename = "final_test_results_detailed.csv"
load_dir = Config.OUTPUT_DIR if 'Config' in locals() else ""
file_path = os.path.join(load_dir, filename)

if not os.path.exists(file_path):
    if os.path.exists(filename):
        file_path = filename
    else:
        print(f"❌ Error: Could not find {filename}")
        file_path = None

if file_path:
    df = pd.read_csv(file_path)
    print(f"✅ Loaded {len(df)} sequence predictions from {file_path}")

    conditions = [
        (df['True_Label'] == 1) & (df['Predicted_Label'] == 1),
        (df['True_Label'] == 0) & (df['Predicted_Label'] == 1),
        (df['True_Label'] == 1) & (df['Predicted_Label'] == 0),
        (df['True_Label'] == 0) & (df['Predicted_Label'] == 0)
    ]
    choices = ['TP', 'FP', 'FN', 'TN']
    df['Result_Type'] = np.select(conditions, choices, default='ERR')
    
    save_path = os.path.join(load_dir, "final_test_results_annotated.csv")
    df.to_csv(save_path, index=False)

    counts = df['Result_Type'].value_counts()
    print("\n📊 CONFUSION MATRIX COUNTS:")
    print(f"   True Positives (TP): {counts.get('TP', 0)}")
    print(f"   False Positives (FP): {counts.get('FP', 0)}")
    print(f"   False Negatives (FN): {counts.get('FN', 0)}")
    print(f"   True Negatives (TN): {counts.get('TN', 0)}")

    tp_df = df[df['Result_Type'] == 'TP']
    width_col = 'BP_Region_Width' if 'BP_Region_Width' in df.columns else 'Attention_Width'
    
    if len(tp_df) > 0 and width_col in tp_df.columns:
        widths = tp_df[width_col]
        median_w = widths.median()
        mean_w = widths.mean()
        sharp_count = (widths < 500).sum()  # Now in BP (global coordinates)
        
        print(f"\n📏 BREAKPOINT LOCALIZATION STATS (TP Only, Global BP):")
        print(f"   Median Width: {median_w:.1f} bp")
        print(f"   Mean Width:   {mean_w:.1f} bp")
        print(f"   Sharp Signals (<500bp): {sharp_count} / {len(tp_df)} ({sharp_count/len(tp_df)*100:.1f}%)")

        plt.figure(figsize=(10, 5))
        plt.hist(widths, bins=50, color='purple', alpha=0.7)
        plt.axvline(median_w, color='black', linestyle='--', label=f'Median: {median_w:.0f}bp')
        plt.title("Distribution of Predicted Breakpoint Widths (TP)", fontsize=14)
        plt.xlabel("Width (bp, global)")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    prob_col = 'Fusion_Prob' if 'Fusion_Prob' in df.columns else 'Prob_Fusion'
    correct_mask = (df['Result_Type'] == 'TP') | (df['Result_Type'] == 'TN')
    
    if prob_col in df.columns:
        df['Confidence'] = (df[prob_col] - 0.5).abs() * 2
        
        avg_conf_correct = df[correct_mask]['Confidence'].mean()
        avg_conf_wrong = df[~correct_mask]['Confidence'].mean()
        
        print("\n🧠 MODEL CERTAINTY:")
        print(f"   Avg Confidence on Correct Preds: {avg_conf_correct:.4f}")
        print(f"   Avg Confidence on Errors:        {avg_conf_wrong:.4f}")


# ==============================================================================
# CELL 10: WINDOW-LEVEL ANALYSIS (NEW)
# ==============================================================================
def analyze_window_contributions(model, loader, device, n_examples=5):
    """
    Analyze how different windows contribute to sequence predictions.
    Visualizes attention patterns across windows for selected sequences.
    """
    print("\n📊 WINDOW CONTRIBUTION ANALYSIS")
    print("="*50)
    
    model.eval()
    
    # Collect window data
    window_data = defaultdict(list)
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Collecting window data"):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            seq_idx = batch['seq_idx'].cpu().numpy()
            win_start = batch['window_start'].cpu().numpy()
            lbls = batch['labels'].cpu().numpy()
            
            outputs = model(ids, attention_mask=mask)
            probs = torch.softmax(outputs['logits'], dim=1).cpu().numpy()
            max_attn = outputs['max_attention'].cpu().numpy()
            bp_idx = outputs['breakpoint_index'].cpu().numpy()
            
            for i in range(len(seq_idx)):
                window_data[seq_idx[i]].append({
                    'window_start': win_start[i],
                    'fusion_prob': probs[i, 1],
                    'max_attention': max_attn[i],
                    'bp_token': bp_idx[i],
                    'label': lbls[i]
                })
    
    # Find interesting sequences (high variance in window predictions)
    seq_variance = {}
    for seq_idx, windows in window_data.items():
        probs = [w['fusion_prob'] for w in windows]
        seq_variance[seq_idx] = np.var(probs)
    
    # Sort by variance
    sorted_seqs = sorted(seq_variance.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n🔍 Top {n_examples} sequences with highest window variance:")
    
    fig, axes = plt.subplots(n_examples, 1, figsize=(14, 4*n_examples))
    if n_examples == 1:
        axes = [axes]
    
    for i, (seq_idx, variance) in enumerate(sorted_seqs[:n_examples]):
        windows = sorted(window_data[seq_idx], key=lambda x: x['window_start'])
        
        starts = [w['window_start'] for w in windows]
        probs = [w['fusion_prob'] for w in windows]
        attns = [w['max_attention'] for w in windows]
        label = windows[0]['label']
        
        ax = axes[i]
        
        # Plot fusion probability per window
        ax.bar(range(len(windows)), probs, alpha=0.7, color='blue', label='Fusion Prob')
        ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Threshold')
        
        # Highlight max
        max_idx = np.argmax(probs)
        ax.bar(max_idx, probs[max_idx], color='green', alpha=0.9, label=f'Best Window')
        
        ax.set_xlabel('Window Index')
        ax.set_ylabel('Fusion Probability')
        ax.set_title(f'Seq {seq_idx} | True Label: {label} | Variance: {variance:.4f} | Max Prob: {max(probs):.3f}')
        ax.legend(loc='upper right')
        ax.set_ylim(0, 1.1)
        
        # Add window start positions as secondary x-axis labels
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(range(len(windows)))
        ax2.set_xticklabels([f'{s//1000}k' for s in starts], fontsize=8)
        ax2.set_xlabel('Window Start (bp)')
    
    plt.tight_layout()
    plt.show()
    
    # Summary statistics
    print("\n📈 WINDOW STATISTICS SUMMARY:")
    all_variances = list(seq_variance.values())
    print(f"   Mean variance across sequences: {np.mean(all_variances):.4f}")
    print(f"   Max variance: {np.max(all_variances):.4f}")
    print(f"   Sequences with high variance (>0.1): {sum(1 for v in all_variances if v > 0.1)}")

# Run analysis if test_loader exists
if 'test_loader' in locals():
    analyze_window_contributions(model, test_loader, device, n_examples=5)


# ==============================================================================
# CELL 11: GEOMETRIC VISUAL VERIFICATION (GLOBAL COORDINATES)
# ==============================================================================
def visualize_breakpoint_landscape(model, loader, device, dataset, n_examples=3):
    """
    Creates a full-sequence view showing attention peaks across all windows.
    Maps everything back to global genomic coordinates.
    """
    print("\n🗺️ BREAKPOINT LANDSCAPE VISUALIZATION")
    print("="*50)
    
    model.eval()
    
    # Collect all window attention data
    seq_attention_data = defaultdict(list)
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Collecting attention maps"):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            seq_idx = batch['seq_idx'].cpu().numpy()
            win_start = batch['window_start'].cpu().numpy()
            lbls = batch['labels'].cpu().numpy()
            
            outputs = model(ids, attention_mask=mask)
            attn_maps = outputs['attention_probs'].cpu().numpy()
            probs = torch.softmax(outputs['logits'], dim=1).cpu().numpy()
            
            for i in range(len(seq_idx)):
                seq_attention_data[seq_idx[i]].append({
                    'window_start': win_start[i],
                    'attention': attn_maps[i],
                    'fusion_prob': probs[i, 1],
                    'label': lbls[i]
                })
    
    # Find positive sequences with high confidence
    positive_seqs = []
    for seq_idx, windows in seq_attention_data.items():
        max_prob = max(w['fusion_prob'] for w in windows)
        label = windows[0]['label']
        if label == 1 and max_prob > 0.8:
            positive_seqs.append((seq_idx, max_prob))
    
    positive_seqs.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nVisualizing top {n_examples} high-confidence positive sequences...")
    
    fig, axes = plt.subplots(n_examples, 1, figsize=(16, 5*n_examples))
    if n_examples == 1:
        axes = [axes]
    
    for i, (seq_idx, max_prob) in enumerate(positive_seqs[:n_examples]):
        ax = axes[i]
        windows = seq_attention_data[seq_idx]
        
        # Create global attention landscape
        global_len = Config.ORIGINAL_SEQ_LEN
        global_attention = np.zeros(global_len)
        global_counts = np.zeros(global_len)  # For averaging overlaps
        
        for w in windows:
            win_start = w['window_start']
            attn = w['attention']
            
            # Map token attention to bp coordinates
            for tok_idx, attn_val in enumerate(attn):
                # Each token represents ~6bp
                bp_start = win_start + tok_idx * 6
                bp_end = min(bp_start + 6, global_len)
                
                if bp_start < global_len:
                    global_attention[bp_start:bp_end] += attn_val
                    global_counts[bp_start:bp_end] += 1
        
        # Average overlapping regions
        global_counts[global_counts == 0] = 1
        global_attention = global_attention / global_counts
        
        # Smooth for visualization
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(global_attention, sigma=50)
        
        # Plot
        x = np.arange(global_len)
        ax.fill_between(x, smoothed, alpha=0.3, color='blue')
        ax.plot(x, smoothed, color='blue', linewidth=0.5, label='Attention (smoothed)')
        
        # Mark peak
        peak_idx = np.argmax(smoothed)
        ax.axvline(peak_idx, color='red', linestyle='--', linewidth=2, 
                   label=f'Peak: {peak_idx:,} bp')
        
        # Mark window boundaries
        for w in windows:
            ax.axvline(w['window_start'], color='gray', alpha=0.2, linewidth=0.5)
        
        # Mark expected breakpoint region (middle of sequence)
        expected_bp = global_len // 2
        ax.axvline(expected_bp, color='green', linestyle=':', linewidth=2,
                   label=f'Expected BP: {expected_bp:,} bp')
        
        ax.set_xlabel('Genomic Position (bp)')
        ax.set_ylabel('Attention')
        ax.set_title(f'Sequence {seq_idx} | Label: Positive | Max Prob: {max_prob:.3f}')
        ax.legend(loc='upper right')
        ax.set_xlim(0, global_len)
        
        # Add scale bar
        ax.text(0.02, 0.95, f'Total: {global_len:,} bp', transform=ax.transAxes,
                fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    plt.show()

# Run visualization
if 'test_loader' in locals() and 'test_dataset' in locals():
    try:
        visualize_breakpoint_landscape(model, test_loader, device, test_dataset, n_examples=3)
    except Exception as e:
        print(f"⚠️ Visualization error: {e}")


# ==============================================================================
# CELL 12: CLINICAL REALITY PR CURVE
# ==============================================================================
from sklearn.metrics import precision_recall_curve, auc

def plot_clinical_impact(y_true, y_probs, clinical_prevalence=0.01):
    precision_bal, recall_bal, _ = precision_recall_curve(y_true, y_probs)
    auc_bal = auc(recall_bal, precision_bal)
    
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    neg_weight = (n_pos / n_neg) * ((1 - clinical_prevalence) / clinical_prevalence)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        fp_ratio = (1 / precision_bal) - 1
        precision_clin = 1 / (1 + fp_ratio * neg_weight)
    
    precision_clin = np.nan_to_num(precision_clin, nan=0.0)
    auc_clin = auc(recall_bal, precision_clin)

    print("-" * 40)
    print(f"📊 CLINICAL METRICS REPORT (Prevalence={clinical_prevalence*100}%)")
    print("-" * 40)
    print(f"1. Clinical AUPRC:      {auc_clin:.4f}")
    
    idx_20 = (np.abs(recall_bal - 0.20)).argmin()
    prec_at_20 = precision_clin[idx_20]
    print(f"2. Precision at 20% Recall: {prec_at_20:.4f}")
    
    high_conf_indices = np.where(precision_clin >= 0.90)[0]
    if len(high_conf_indices) > 0:
        rec_at_90 = recall_bal[high_conf_indices].max()
        print(f"3. Recall at 90% Precision: {rec_at_90:.4f}")
    else:
        print("3. Recall at 90% Precision: 0.0000")

    break_even_indices = np.where(precision_clin >= 0.50)[0]
    if len(break_even_indices) > 0:
        rec_at_50 = recall_bal[break_even_indices].max()
        print(f"4. Drop-off (Recall @ 50% Prec): {rec_at_50:.4f}")
    else:
        print("4. Drop-off (Recall @ 50% Prec): 0.0000")
    print("-" * 40)

    plt.figure(figsize=(10, 7))
    plt.plot(recall_bal, precision_bal, color='blue', lw=2, alpha=0.6, 
             label=f'Balanced Test Set (AUC={auc_bal:.3f})')
    plt.plot(recall_bal, precision_clin, color='red', lw=3, 
             label=f'Clinical Scenario (1% Prev) (AUC={auc_clin:.3f})')
    
    plt.axhline(y=clinical_prevalence, color='red', linestyle=':', label='Clinical Baseline (1%)')
    plt.axhline(y=0.5, color='blue', linestyle=':', label='Balanced Baseline (50%)')
    
    plt.title("Clinical Utility Analysis (PR Curve) - Sliding Window", fontsize=14)
    plt.xlabel("Recall (Sensitivity)", fontsize=12)
    plt.ylabel("Precision (PPV)", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.05])
    plt.show()

if 'test_true' in locals() and 'test_probs' in locals():
    plot_clinical_impact(test_true, test_probs[:, 1], clinical_prevalence=0.01)
else:
    print("⚠️ Error: Run test evaluation first.")


# ==============================================================================
# CELL 13: EXPORT SUMMARY & MODEL CARD
# ==============================================================================
print("\n" + "="*60)
print("📋 SLIDING WINDOW NUCLEOTIDE TRANSFORMER - SUMMARY")
print("="*60)

print(f"""
MODEL CONFIGURATION:
  - Base Model: {Config.MODEL_NAME}
  - Window Size: {Config.WINDOW_BP:,} bp
  - Stride: {Config.STRIDE_BP:,} bp ({Config.STRIDE_BP/Config.WINDOW_BP*100:.0f}% overlap)
  - Max Tokens: {Config.MAX_TOKENS}
  - Original Sequence Length: {Config.ORIGINAL_SEQ_LEN:,} bp
  - Aggregation Strategy: {Config.AGGREGATION_STRATEGY}

TRAINING:
  - Epochs: {Config.EPOCHS}
  - Batch Size: {Config.BATCH_SIZE}
  - Learning Rate: {Config.LEARNING_RATE}

KEY FEATURES:
  ✓ Sliding window for full 32k coverage
  ✓ Attention-based pooling preserved
  ✓ Global breakpoint coordinate mapping
  ✓ Window-level and sequence-level metrics
  ✓ Multiple aggregation strategies available

OUTPUT FILES:
  - {Config.OUTPUT_DIR}/best_model.safetensors
  - {Config.OUTPUT_DIR}/final_test_results_detailed.csv
""")
