# ==============================================================================
# FUSIONAI FAIR BENCHMARK PIPELINE (RAW FILE LOADING VERSION)
# ------------------------------------------------------------------------------
# This script:
# 1) Loads the RAW files directly:
#       POSITIVE_CSV
#       NEGATIVE_CSV
#       TEST_CSV
#       EXTRA_POS_FASTA
#       NEG_FASTA_CANONICAL
#       NEG_FASTA_SYNTHETIC
# 2) Rebuilds the benchmark dataset with the SAME jitter logic used by FuseLens
# 3) Uses a strict biological group split (same policy as FuseLens)
# 4) Trains a FusionAI-style CNN adapted to the same 32k windows
# 5) Evaluates on the internal blind test split
# 6) Evaluates separately on TEST_CSV as an external positive-only set
# 7) Optionally compares against FuseLens predictions if available
# ==============================================================================

# ==============================================================================
# CELL 1: IMPORTS & CONFIG
# ==============================================================================
import os
import json
import random
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
)

import matplotlib.pyplot as plt
import seaborn as sns


# ------------------------------------------------------------------------------
# Fallback Config if FuseLens Config is not already defined in the notebook
# ------------------------------------------------------------------------------
if "Config" not in globals():
    class Config:
        POSITIVE_CSV = "datasets/fusion_gene_positive_bp_information_with_class_for_modeling.txt"
        NEGATIVE_CSV = "datasets/fusion_gene_negative_bp_information_with_class_for_modeling.txt"
        TEST_CSV = "datasets/fusion_gene_positive_bp_information_with_class_for_testing.txt"

        EXTRA_POS_FASTA = "datasets/blast_validated_chimeras.fasta,datasets/cosmic_high_confidence_sequences.fna,datasets/chimeras_43466.fa"
        NEG_FASTA_CANONICAL = "datasets/false_negative_candidates.fasta"
        NEG_FASTA_SYNTHETIC = "datasets/false_positive_candidates.fasta"

        OUTPUT_DIR = "./hyenadna_v9.9_checkpoints_32k"
        MODEL_NAME = "LongSafari/hyenadna-small-32k-seqlen-hf"

        MAX_LEN = 32768
        BATCH_SIZE = 8
        EPOCHS = 3
        LEARNING_RATE = 1e-5
        VAL_SPLIT = 0.2
        SEED = 42
        CONFIDENCE_THRESHOLD = 0.90


class FusionAIBenchmarkConfig:
    # Use same raw files and same maximum length as FuseLens
    POSITIVE_CSV = Config.POSITIVE_CSV
    NEGATIVE_CSV = Config.NEGATIVE_CSV
    TEST_CSV = Config.TEST_CSV

    EXTRA_POS_FASTA = Config.EXTRA_POS_FASTA
    NEG_FASTA_CANONICAL = Config.NEG_FASTA_CANONICAL
    NEG_FASTA_SYNTHETIC = Config.NEG_FASTA_SYNTHETIC

    MAX_LEN = Config.MAX_LEN
    SEED = Config.SEED

    # Model / training
    BATCH_SIZE = 8               # reduce to 4 if you hit OOM
    MAX_EPOCHS = 15
    EARLY_STOPPING_PATIENCE = 3

    # FusionAI-style optimizer choice
    LEARNING_RATE = 1.0
    ADADELTA_RHO = 0.95
    ADADELTA_EPS = 1e-7
    WEIGHT_DECAY = 0.0

    # Dropout values close to released FusionAI config
    DROPOUT_1 = 0.25
    DROPOUT_2 = 0.40

    NUM_WORKERS = min(4, os.cpu_count() or 1)
    PIN_MEMORY = torch.cuda.is_available()
    USE_AMP = torch.cuda.is_available()

    INTERNAL_TEST_THRESHOLD = 0.50
    OUTPUT_DIR = os.path.join(Config.OUTPUT_DIR, "fusionai_rawfile_benchmark")


os.makedirs(FusionAIBenchmarkConfig.OUTPUT_DIR, exist_ok=True)


def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_all_seeds(FusionAIBenchmarkConfig.SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ FusionAI benchmark setup complete.")
print(f"   Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")


def seed_worker(worker_id):
    worker_seed = FusionAIBenchmarkConfig.SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ==============================================================================
# CELL 2: DATA PREPARATION UTILITIES
# ==============================================================================
class DataPreparator:
    COLUMNS = [
        "Hgene", "Hchr", "Hbp", "Hstrand",
        "Tgene", "Tchr", "Tbp", "Tstrand",
        "5'-gene sequence (10Kb)", "3'-gene sequence (10Kb)"
    ]

    def __init__(self, config):
        self.cfg = config
        self.trans_table = str.maketrans("ATCGN", "TAGCN")

    def _trim_artifacts(self, sequence: str) -> str:
        if not isinstance(sequence, str):
            return ""
        return sequence.upper().strip()

    def _get_reverse_complement(self, sequence: str) -> str:
        return sequence.upper().translate(self.trans_table)[::-1]

    def _generate_random_dna(self, length: int) -> str:
        if length <= 0:
            return ""
        return "".join(random.choices("ACGT", k=length))

    def _apply_jitter(self, seq_5p: str, seq_3p: str) -> str:
        """
        SAME logic as FuseLens:
        - combine 5' + 3'
        - if shorter than MAX_LEN: random left/right ACGT padding
        - if longer than MAX_LEN: random crop
        """
        seq_5p = self._trim_artifacts(seq_5p)
        seq_3p = self._trim_artifacts(seq_3p)

        core_seq = seq_5p + seq_3p
        core_len = len(core_seq)
        target_len = self.cfg.MAX_LEN

        if core_len < target_len:
            needed = target_len - core_len
            pad_left = random.randint(0, needed)
            pad_right = needed - pad_left

            left_seq = self._generate_random_dna(pad_left)
            right_seq = self._generate_random_dna(pad_right)
            return left_seq + core_seq + right_seq

        max_start = core_len - target_len
        start = random.randint(0, max_start)
        return core_seq[start:start + target_len]

    def _load_fasta(self, path: str) -> list:
        if not os.path.exists(path):
            return []
        seqs, curr = [], []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if curr:
                        seqs.append(self._trim_artifacts("".join(curr)))
                    curr = []
                else:
                    curr.append(line)
            if curr:
                seqs.append(self._trim_artifacts("".join(curr)))
        return seqs


def md5_text(x: str) -> str:
    return hashlib.md5(str(x).encode("utf-8")).hexdigest()


def add_sample_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sample_key"] = df["sequence"].astype(str).apply(md5_text)
    return df


def print_length_stats(name: str, lengths: list):
    if len(lengths) == 0:
        print(f"{name}: no sequences")
        return
    arr = np.asarray(lengths)
    print(f"{name}:")
    print(f"  Count: {len(arr):,}")
    print(f"  Mean:  {arr.mean():.2f}")
    print(f"  Median:{np.median(arr):.2f}")
    print(f"  Min:   {arr.min()}")
    print(f"  Max:   {arr.max()}")


# ==============================================================================
# CELL 3: RAW FILE LOADING
# ==============================================================================
preparator = DataPreparator(FusionAIBenchmarkConfig)

all_pos_lengths = []
all_neg_lengths = []
all_ext_lengths = []


def load_positive_modeling_pool(preparator: DataPreparator) -> pd.DataFrame:
    pos_dfs = []

    # --------------------------------------------------------------------------
    # 1. Positive CSV (modeling)
    # --------------------------------------------------------------------------
    if os.path.exists(FusionAIBenchmarkConfig.POSITIVE_CSV):
        print(f"\n📂 Loading positive modeling CSV:")
        print(f"   {FusionAIBenchmarkConfig.POSITIVE_CSV}")

        csv_df = pd.read_csv(
            FusionAIBenchmarkConfig.POSITIVE_CSV,
            header=None,
            names=DataPreparator.COLUMNS,
            sep="\t"
        )

        raw_5 = csv_df["5'-gene sequence (10Kb)"].astype(str).str.strip()
        raw_3 = csv_df["3'-gene sequence (10Kb)"].astype(str).str.strip()
        raw_concat = raw_5 + raw_3

        raw_lens = raw_5.str.len() + raw_3.str.len()
        all_pos_lengths.extend(raw_lens.tolist())

        csv_df["sequence"] = csv_df.apply(
            lambda r: preparator._apply_jitter(
                r["5'-gene sequence (10Kb)"],
                r["3'-gene sequence (10Kb)"]
            ),
            axis=1
        )
        csv_df["label"] = 1
        csv_df["source_type"] = "positive_csv_modeling"
        csv_df["source_file"] = os.path.basename(FusionAIBenchmarkConfig.POSITIVE_CSV)
        csv_df["fusion_name"] = csv_df["Hgene"].astype(str) + "--" + csv_df["Tgene"].astype(str)
        csv_df["gene_name"] = pd.NA
        csv_df["raw_hash"] = raw_concat.apply(md5_text)

        pos_dfs.append(csv_df)
        print(f"   Loaded {len(csv_df):,} positive CSV rows.")
    else:
        print(f"⚠️ Positive CSV not found: {FusionAIBenchmarkConfig.POSITIVE_CSV}")

    # --------------------------------------------------------------------------
    # 2. Extra positive FASTA files
    # --------------------------------------------------------------------------
    extra_paths = [p.strip() for p in FusionAIBenchmarkConfig.EXTRA_POS_FASTA.split(",") if p.strip()]
    for fasta_path in extra_paths:
        if not os.path.exists(fasta_path):
            print(f"⚠️ Extra positive FASTA not found: {fasta_path}")
            continue

        raw_seqs = preparator._load_fasta(fasta_path)
        all_pos_lengths.extend([len(s) for s in raw_seqs])

        processed = [preparator._apply_jitter(s, "") for s in raw_seqs]
        raw_hashes = [md5_text(s) for s in raw_seqs]

        fasta_df = pd.DataFrame({
            "sequence": processed,
            "label": 1,
            "source_type": "positive_fasta_extra",
            "source_file": os.path.basename(fasta_path),
            "fusion_name": pd.NA,
            "gene_name": pd.NA,
            "raw_hash": raw_hashes,
        })
        pos_dfs.append(fasta_df)
        print(f"   Loaded {len(fasta_df):,} extra positives from {os.path.basename(fasta_path)}")

    if len(pos_dfs) == 0:
        return pd.DataFrame(columns=["sequence", "label", "source_type", "source_file", "fusion_name", "gene_name", "raw_hash", "sample_key"])

    pos_df = pd.concat(pos_dfs, ignore_index=True)
    pos_df.drop_duplicates(subset=["sequence"], inplace=True)
    pos_df = add_sample_key(pos_df)
    pos_df.reset_index(drop=True, inplace=True)
    return pos_df


def load_negative_modeling_pool(preparator: DataPreparator) -> pd.DataFrame:
    neg_dfs = []

    # --------------------------------------------------------------------------
    # 1. Negative CSV
    # --------------------------------------------------------------------------
    if os.path.exists(FusionAIBenchmarkConfig.NEGATIVE_CSV):
        print(f"\n📂 Loading negative modeling CSV:")
        print(f"   {FusionAIBenchmarkConfig.NEGATIVE_CSV}")

        neg_csv = pd.read_csv(
            FusionAIBenchmarkConfig.NEGATIVE_CSV,
            header=None,
            names=DataPreparator.COLUMNS,
            sep="\t"
        )

        raw_5 = neg_csv["5'-gene sequence (10Kb)"].astype(str).str.strip()
        raw_3 = neg_csv["3'-gene sequence (10Kb)"].astype(str).str.strip()
        raw_concat = raw_5 + raw_3

        raw_lens = raw_5.str.len() + raw_3.str.len()
        all_neg_lengths.extend(raw_lens.tolist())

        neg_csv["sequence"] = neg_csv.apply(
            lambda r: preparator._apply_jitter(
                r["5'-gene sequence (10Kb)"],
                r["3'-gene sequence (10Kb)"]
            ),
            axis=1
        )
        neg_csv["label"] = 0
        neg_csv["source_type"] = "negative_csv_modeling"
        neg_csv["source_file"] = os.path.basename(FusionAIBenchmarkConfig.NEGATIVE_CSV)
        neg_csv["fusion_name"] = pd.NA
        neg_csv["gene_name"] = neg_csv["Hgene"].astype(str)
        neg_csv["raw_hash"] = raw_concat.apply(md5_text)

        # EXACTLY LIKE FUSELENS: reverse-complement augment the negative CSV
        print("   Applying reverse-complement augmentation to negative CSV...")
        neg_aug = neg_csv.copy()
        neg_aug["sequence"] = neg_aug["sequence"].apply(preparator._get_reverse_complement)
        neg_aug["source_type"] = "negative_csv_modeling_rc"

        neg_csv_full = pd.concat([neg_csv, neg_aug], ignore_index=True)
        neg_dfs.append(neg_csv_full)

        print(f"   Loaded {len(neg_csv):,} raw negative CSV rows -> {len(neg_csv_full):,} after RC augmentation.")
    else:
        print(f"⚠️ Negative CSV not found: {FusionAIBenchmarkConfig.NEGATIVE_CSV}")

    # --------------------------------------------------------------------------
    # 2. Negative FASTA sources
    # --------------------------------------------------------------------------
    neg_fasta_sources = [
        FusionAIBenchmarkConfig.NEG_FASTA_CANONICAL,
        FusionAIBenchmarkConfig.NEG_FASTA_SYNTHETIC,
    ]

    for fasta_path in neg_fasta_sources:
        if not os.path.exists(fasta_path):
            print(f"⚠️ Negative FASTA not found: {fasta_path}")
            continue

        raw_negs = preparator._load_fasta(fasta_path)
        all_neg_lengths.extend([len(s) for s in raw_negs])

        processed_negs = [preparator._apply_jitter(s, "") for s in raw_negs]
        raw_hashes = [md5_text(s) for s in raw_negs]

        fasta_df = pd.DataFrame({
            "sequence": processed_negs,
            "label": 0,
            "source_type": "negative_fasta",
            "source_file": os.path.basename(fasta_path),
            "fusion_name": pd.NA,
            "gene_name": pd.NA,
            "raw_hash": raw_hashes,
        })
        neg_dfs.append(fasta_df)
        print(f"   Loaded {len(fasta_df):,} negatives from {os.path.basename(fasta_path)}")

    if len(neg_dfs) == 0:
        return pd.DataFrame(columns=["sequence", "label", "source_type", "source_file", "fusion_name", "gene_name", "raw_hash", "sample_key"])

    neg_df = pd.concat(neg_dfs, ignore_index=True)
    neg_df.drop_duplicates(subset=["sequence"], inplace=True)
    neg_df = add_sample_key(neg_df)
    neg_df.reset_index(drop=True, inplace=True)
    return neg_df


def load_external_positive_test_csv(preparator: DataPreparator) -> pd.DataFrame:
    """
    Load TEST_CSV separately.
    In your current file naming, this appears to be an additional positive-only test file.
    We keep it separate from training/validation/internal blind test to avoid leakage.
    """
    if not os.path.exists(FusionAIBenchmarkConfig.TEST_CSV):
        print(f"⚠️ External TEST_CSV not found: {FusionAIBenchmarkConfig.TEST_CSV}")
        return pd.DataFrame(columns=["sequence", "label", "source_type", "source_file", "fusion_name", "gene_name", "raw_hash", "sample_key"])

    print(f"\n📂 Loading external TEST_CSV:")
    print(f"   {FusionAIBenchmarkConfig.TEST_CSV}")

    ext_df = pd.read_csv(
        FusionAIBenchmarkConfig.TEST_CSV,
        header=None,
        names=DataPreparator.COLUMNS,
        sep="\t"
    )

    raw_5 = ext_df["5'-gene sequence (10Kb)"].astype(str).str.strip()
    raw_3 = ext_df["3'-gene sequence (10Kb)"].astype(str).str.strip()
    raw_concat = raw_5 + raw_3

    raw_lens = raw_5.str.len() + raw_3.str.len()
    all_ext_lengths.extend(raw_lens.tolist())

    ext_df["sequence"] = ext_df.apply(
        lambda r: preparator._apply_jitter(
            r["5'-gene sequence (10Kb)"],
            r["3'-gene sequence (10Kb)"]
        ),
        axis=1
    )
    ext_df["label"] = 1
    ext_df["source_type"] = "external_positive_test_csv"
    ext_df["source_file"] = os.path.basename(FusionAIBenchmarkConfig.TEST_CSV)
    ext_df["fusion_name"] = ext_df["Hgene"].astype(str) + "--" + ext_df["Tgene"].astype(str)
    ext_df["gene_name"] = pd.NA
    ext_df["raw_hash"] = raw_concat.apply(md5_text)

    ext_df.drop_duplicates(subset=["sequence"], inplace=True)
    ext_df = add_sample_key(ext_df)
    ext_df.reset_index(drop=True, inplace=True)

    print(f"   Loaded {len(ext_df):,} external positive test rows.")
    return ext_df


# Execute raw loading
pos_df_fai = load_positive_modeling_pool(preparator)
neg_df_fai = load_negative_modeling_pool(preparator)
ext_test_pos_df_fai = load_external_positive_test_csv(preparator)

print("\n📊 LENGTH STATS")
print_length_stats("Positives (raw before jitter)", all_pos_lengths)
print_length_stats("Negatives (raw before jitter)", all_neg_lengths)
print_length_stats("External TEST_CSV positives (raw before jitter)", all_ext_lengths)

print("\n📦 FINAL MODELING POOL AFTER PROCESSING / DEDUP")
print(f"   Positives: {len(pos_df_fai):,}")
print(f"   Negatives: {len(neg_df_fai):,}")

# ==============================================================================
# CELL 4: STRICT BIOLOGICAL SPLIT
# ==============================================================================
print("\n🔬 Performing strict biological split...")

full_df_fai = pd.concat([pos_df_fai, neg_df_fai], ignore_index=True).copy()
full_df_fai.drop_duplicates(subset=["sequence"], inplace=True)
full_df_fai.reset_index(drop=True, inplace=True)

def get_biological_group_id(row):
    # Match FuseLens splitting philosophy as closely as possible
    if row["label"] == 1 and pd.notna(row.get("fusion_name", pd.NA)):
        return f"POS::{row['fusion_name']}"
    if row["label"] == 0 and pd.notna(row.get("gene_name", pd.NA)):
        return f"NEG::{row['gene_name']}"
    if pd.notna(row.get("raw_hash", pd.NA)):
        return f"RAW::{row['label']}::{row['source_file']}::{row['raw_hash']}"
    return f"SEQ::{row['label']}::{row['sample_key']}"

full_df_fai["split_group_id"] = full_df_fai.apply(get_biological_group_id, axis=1)

unique_groups = full_df_fai["split_group_id"].unique()
train_groups, temp_groups = train_test_split(
    unique_groups,
    test_size=0.30,
    random_state=FusionAIBenchmarkConfig.SEED
)
val_groups, test_groups = train_test_split(
    temp_groups,
    test_size=0.50,
    random_state=FusionAIBenchmarkConfig.SEED
)

train_df_fai = full_df_fai[full_df_fai["split_group_id"].isin(train_groups)].copy().reset_index(drop=True)
val_df_fai   = full_df_fai[full_df_fai["split_group_id"].isin(val_groups)].copy().reset_index(drop=True)
test_df_fai  = full_df_fai[full_df_fai["split_group_id"].isin(test_groups)].copy().reset_index(drop=True)

overlap_train_test = set(train_df_fai["split_group_id"]).intersection(set(test_df_fai["split_group_id"]))
overlap_train_val  = set(train_df_fai["split_group_id"]).intersection(set(val_df_fai["split_group_id"]))
overlap_val_test   = set(val_df_fai["split_group_id"]).intersection(set(test_df_fai["split_group_id"]))

print(f"   Total unique samples:   {len(full_df_fai):,}")
print(f"   Unique biological groups: {len(unique_groups):,}")
print(f"   Train: {len(train_df_fai):,}")
print(f"   Val:   {len(val_df_fai):,}")
print(f"   Test:  {len(test_df_fai):,}")

if len(overlap_train_test) == 0 and len(overlap_train_val) == 0 and len(overlap_val_test) == 0:
    print("✅ Leakage check passed: no biological groups are shared across splits.")
else:
    print("❌ Leakage detected!")
    print(f"   Train/Test overlap: {len(overlap_train_test)}")
    print(f"   Train/Val overlap:  {len(overlap_train_val)}")
    print(f"   Val/Test overlap:   {len(overlap_val_test)}")

# Save the exact benchmark splits
train_df_fai.to_csv(os.path.join(FusionAIBenchmarkConfig.OUTPUT_DIR, "fusionai_train_split.csv"), index=False)
val_df_fai.to_csv(os.path.join(FusionAIBenchmarkConfig.OUTPUT_DIR, "fusionai_val_split.csv"), index=False)
test_df_fai.to_csv(os.path.join(FusionAIBenchmarkConfig.OUTPUT_DIR, "fusionai_test_split.csv"), index=False)

# ==============================================================================
# CELL 5: DATASET CLASS (FUSIONAI INPUT ENCODING)
# ==============================================================================
DNA_LUT = np.full(256, 4, dtype=np.uint8)  # default unknown/N
DNA_LUT[ord("A")] = 0
DNA_LUT[ord("C")] = 1
DNA_LUT[ord("G")] = 2
DNA_LUT[ord("T")] = 3
DNA_LUT[ord("N")] = 4


class FusionAISequenceDataset(Dataset):
    """
    Expects sequences to already be fixed-length benchmark windows.
    Encoding:
      A=0, C=1, G=2, T=3, N/unknown=4
    Inside the model, this becomes 4-channel one-hot with N -> all-zero row.
    """
    def __init__(self, df: pd.DataFrame, input_len: int):
        self.df = df.reset_index(drop=True).copy()
        self.input_len = input_len
        self.sequences = self.df["sequence"].astype(str).tolist()
        self.labels = self.df["label"].astype(int).tolist()
        self.sample_keys = self.df["sample_key"].astype(str).tolist()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        if len(seq) != self.input_len:
            raise ValueError(
                f"Sequence at idx={idx} has length {len(seq)} but expected {self.input_len}. "
                "For a fair benchmark, sequences must already be final fixed-length windows."
            )

        arr = np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)
        arr = DNA_LUT[arr].astype(np.int64)

        return {
            "input_ids": torch.from_numpy(arr),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "sample_key": self.sample_keys[idx],
        }


# ==============================================================================
# CELL 6: FUSIONAI-STYLE MODEL
# ==============================================================================
class FusionAI32kModel(nn.Module):
    """
    FusionAI-style architecture adapted to the same 32k benchmark windows.

    Input:
      integer-coded sequence of shape (B, L)
    Internal transform:
      4-channel one-hot (A/C/G/T), N -> zeros
      reshaped to (B, 1, L, 4)

    Architecture:
      Conv2d(1 -> 256, kernel=(20,4))
      Conv2d(256 -> 32, kernel=(200,1))
      MaxPool2d((20,1))
      Dropout(0.25)
      Flatten
      Dense(32)
      Dropout(0.40)
      Dense(2)
    """
    def __init__(self, input_len: int, dropout1=0.25, dropout2=0.40):
        super().__init__()
        self.input_len = input_len

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=256,
            kernel_size=(20, 4),
            stride=(1, 1),
            padding=0,
            bias=True,
        )
        self.conv2 = nn.Conv2d(
            in_channels=256,
            out_channels=32,
            kernel_size=(200, 1),
            stride=(1, 1),
            padding=0,
            bias=True,
        )
        self.pool = nn.MaxPool2d(kernel_size=(20, 1), stride=(20, 1))
        self.dropout1 = nn.Dropout(dropout1)

        flat_dim = self._compute_flat_dim(input_len)
        self.fc1 = nn.Linear(flat_dim, 32)
        self.dropout2 = nn.Dropout(dropout2)
        self.fc2 = nn.Linear(32, 2)

    @staticmethod
    def _out_len_valid(L, kernel, stride=1, padding=0, dilation=1):
        return ((L + 2 * padding - dilation * (kernel - 1) - 1) // stride) + 1

    def _compute_flat_dim(self, input_len):
        h1 = self._out_len_valid(input_len, kernel=20, stride=1, padding=0)
        h2 = self._out_len_valid(h1, kernel=200, stride=1, padding=0)
        h3 = self._out_len_valid(h2, kernel=20, stride=20, padding=0)
        if h3 <= 0:
            raise ValueError(f"Input length {input_len} too short for current architecture.")
        return 32 * h3 * 1

    def forward(self, input_ids):
        # input_ids: (B, L) integers in {0,1,2,3,4}
        valid_mask = (input_ids < 4).unsqueeze(-1)            # (B, L, 1)
        clipped = torch.clamp(input_ids, min=0, max=3)
        x = F.one_hot(clipped, num_classes=4).float()         # (B, L, 4)
        x = x * valid_mask.float()                            # N/unknown -> zero row
        x = x.unsqueeze(1)                                    # (B, 1, L, 4)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        logits = self.fc2(x)
        return logits


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==============================================================================
# CELL 7: METRICS / EVALUATION / PLOTS
# ==============================================================================
def compute_binary_metrics(y_true, pos_probs, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    pos_probs = np.asarray(pos_probs).astype(float)
    y_pred = (pos_probs >= threshold).astype(int)

    out = {
        "threshold": float(threshold),
        "acc": accuracy_score(y_true, y_pred),
        "prec": precision_score(y_true, y_pred, zero_division=0),
        "rec": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }

    try:
        out["auroc"] = roc_auc_score(y_true, pos_probs)
    except ValueError:
        out["auroc"] = float("nan")

    try:
        out["auprc"] = average_precision_score(y_true, pos_probs)
    except ValueError:
        out["auprc"] = float("nan")

    return out


def find_best_threshold(y_true, pos_probs, metric="mcc"):
    thresholds = np.linspace(0.05, 0.95, 181)
    best_thr = 0.50
    best_score = -np.inf

    for thr in thresholds:
        m = compute_binary_metrics(y_true, pos_probs, threshold=thr)
        score = m[metric]
        if score > best_score:
            best_score = score
            best_thr = float(thr)

    return best_thr, float(best_score)


def plot_test_results(y_true, pos_probs, title_prefix="FusionAI"):
    y_true = np.asarray(y_true).astype(int)
    pos_probs = np.asarray(pos_probs).astype(float)
    y_pred = (pos_probs >= 0.5).astype(int)

    plt.figure(figsize=(14, 4.5))

    # A. Confusion Matrix
    plt.subplot(1, 3, 1)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Non-Fusion", "Fusion"],
        yticklabels=["Non-Fusion", "Fusion"],
    )
    plt.title(f"{title_prefix}\nConfusion Matrix")

    # B. ROC Curve
    plt.subplot(1, 3, 2)
    fpr, tpr, _ = roc_curve(y_true, pos_probs)
    plt.plot(fpr, tpr, lw=2, label=f"AUC={roc_auc_score(y_true, pos_probs):.4f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title_prefix}\nROC Curve")
    plt.legend()

    # C. Score Distribution
    plt.subplot(1, 3, 3)
    sns.histplot(pos_probs[y_true == 0], bins=40, color="red", alpha=0.45, stat="density", label="True Neg")
    sns.histplot(pos_probs[y_true == 1], bins=40, color="green", alpha=0.45, stat="density", label="True Pos")
    plt.axvline(0.5, color="black", linestyle="--", lw=1)
    plt.xlabel("Fusion probability")
    plt.title(f"{title_prefix}\nProbability Distribution")
    plt.legend()

    plt.tight_layout()
    plt.show()


@torch.no_grad()
def evaluate_positive_only(model, loader, device, use_amp=False):
    model.eval()
    all_probs = []
    all_keys = []

    for batch in loader:
        ids = batch["input_ids"].to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(ids)
        probs = torch.softmax(logits, dim=1)[:, 1]

        all_probs.extend(probs.cpu().numpy())
        all_keys.extend(batch["sample_key"])

    return np.asarray(all_probs), np.asarray(all_keys)


# ==============================================================================
# CELL 8: TRAINING / EVALUATION LOOPS
# ==============================================================================
def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None, use_amp=False):
    model.train()
    total_loss = 0.0
    all_probs = []
    all_labels = []

    for batch in loader:
        ids = batch["input_ids"].to(device, non_blocking=True)
        lbls = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(ids)
            loss = criterion(logits, lbls)

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        probs = torch.softmax(logits.detach(), dim=1)[:, 1]

        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(lbls.detach().cpu().numpy())

    metrics = compute_binary_metrics(all_labels, all_probs, threshold=FusionAIBenchmarkConfig.INTERNAL_TEST_THRESHOLD)
    metrics["loss"] = total_loss / max(1, len(loader))
    return metrics


@torch.no_grad()
def evaluate_loader(model, loader, criterion, device, use_amp=False):
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []
    all_keys = []

    for batch in loader:
        ids = batch["input_ids"].to(device, non_blocking=True)
        lbls = batch["labels"].to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(ids)
            loss = criterion(logits, lbls)

        total_loss += loss.item()
        probs = torch.softmax(logits, dim=1)[:, 1]

        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(lbls.cpu().numpy())
        all_keys.extend(batch["sample_key"])

    metrics_05 = compute_binary_metrics(all_labels, all_probs, threshold=0.5)
    metrics_05["loss"] = total_loss / max(1, len(loader))

    best_thr, best_score = find_best_threshold(all_labels, all_probs, metric="mcc")
    metrics_opt = compute_binary_metrics(all_labels, all_probs, threshold=best_thr)
    metrics_opt["loss"] = total_loss / max(1, len(loader))
    metrics_opt["best_val_mcc"] = best_score

    return metrics_05, metrics_opt, np.asarray(all_labels), np.asarray(all_probs), np.asarray(all_keys)


# ==============================================================================
# CELL 9: DATASETS / DATALOADERS
# ==============================================================================
train_dataset_fai = FusionAISequenceDataset(train_df_fai, FusionAIBenchmarkConfig.MAX_LEN)
val_dataset_fai   = FusionAISequenceDataset(val_df_fai, FusionAIBenchmarkConfig.MAX_LEN)
test_dataset_fai  = FusionAISequenceDataset(test_df_fai, FusionAIBenchmarkConfig.MAX_LEN)

loader_kwargs = dict(
    num_workers=FusionAIBenchmarkConfig.NUM_WORKERS,
    pin_memory=FusionAIBenchmarkConfig.PIN_MEMORY,
    worker_init_fn=seed_worker,
)

if FusionAIBenchmarkConfig.NUM_WORKERS > 0:
    loader_kwargs["persistent_workers"] = True

train_loader_fai = DataLoader(
    train_dataset_fai,
    batch_size=FusionAIBenchmarkConfig.BATCH_SIZE,
    shuffle=True,
    **loader_kwargs
)

val_loader_fai = DataLoader(
    val_dataset_fai,
    batch_size=FusionAIBenchmarkConfig.BATCH_SIZE,
    shuffle=False,
    **loader_kwargs
)

test_loader_fai = DataLoader(
    test_dataset_fai,
    batch_size=FusionAIBenchmarkConfig.BATCH_SIZE,
    shuffle=False,
    **loader_kwargs
)

# External positive-only set from TEST_CSV
if len(ext_test_pos_df_fai) > 0:
    ext_dataset_fai = FusionAISequenceDataset(ext_test_pos_df_fai, FusionAIBenchmarkConfig.MAX_LEN)
    ext_loader_fai = DataLoader(
        ext_dataset_fai,
        batch_size=FusionAIBenchmarkConfig.BATCH_SIZE,
        shuffle=False,
        **loader_kwargs
    )
else:
    ext_loader_fai = None

print("\n📚 DATALOADERS READY")
print(f"   Train batches: {len(train_loader_fai):,}")
print(f"   Val batches:   {len(val_loader_fai):,}")
print(f"   Test batches:  {len(test_loader_fai):,}")
if ext_loader_fai is not None:
    print(f"   External TEST_CSV batches: {len(ext_loader_fai):,}")

# ==============================================================================
# CELL 10: INITIALIZE MODEL / TRAIN
# ==============================================================================
fusionai_model = FusionAI32kModel(
    input_len=FusionAIBenchmarkConfig.MAX_LEN,
    dropout1=FusionAIBenchmarkConfig.DROPOUT_1,
    dropout2=FusionAIBenchmarkConfig.DROPOUT_2,
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adadelta(
    fusionai_model.parameters(),
    lr=FusionAIBenchmarkConfig.LEARNING_RATE,
    rho=FusionAIBenchmarkConfig.ADADELTA_RHO,
    eps=FusionAIBenchmarkConfig.ADADELTA_EPS,
    weight_decay=FusionAIBenchmarkConfig.WEIGHT_DECAY,
)

# FIXED: PyTorch 2.x updated API for GradScaler
scaler = torch.amp.GradScaler('cuda', enabled=FusionAIBenchmarkConfig.USE_AMP)

print("\n🧠 FusionAI-style model initialized.")
print(f"   Trainable parameters: {count_trainable_params(fusionai_model):,}")

history = []
best_val_auc = -np.inf
best_epoch = -1
best_val_threshold = 0.5
epochs_without_improvement = 0

best_ckpt_path = os.path.join(FusionAIBenchmarkConfig.OUTPUT_DIR, "fusionai_best.pt")
last_ckpt_path = os.path.join(FusionAIBenchmarkConfig.OUTPUT_DIR, "fusionai_last.pt")

print("\n🚀 TRAINING STARTED")
for epoch in range(1, FusionAIBenchmarkConfig.MAX_EPOCHS + 1):
    train_metrics = train_one_epoch(
        fusionai_model,
        train_loader_fai,
        optimizer,
        criterion,
        device,
        scaler=scaler,
        use_amp=FusionAIBenchmarkConfig.USE_AMP,
    )

    val_metrics_05, val_metrics_opt, val_true, val_probs, val_keys = evaluate_loader(
        fusionai_model,
        val_loader_fai,
        criterion,
        device,
        use_amp=FusionAIBenchmarkConfig.USE_AMP,
    )

    row = {
        "epoch": epoch,
        "train_loss": train_metrics["loss"],
        "train_acc": train_metrics["acc"],
        "train_prec": train_metrics["prec"],
        "train_rec": train_metrics["rec"],
        "train_f1": train_metrics["f1"],
        "train_mcc": train_metrics["mcc"],
        "train_auroc": train_metrics["auroc"],
        "train_auprc": train_metrics["auprc"],
        "val_loss": val_metrics_05["loss"],
        "val_acc_05": val_metrics_05["acc"],
        "val_prec_05": val_metrics_05["prec"],
        "val_rec_05": val_metrics_05["rec"],
        "val_f1_05": val_metrics_05["f1"],
        "val_mcc_05": val_metrics_05["mcc"],
        "val_auroc": val_metrics_05["auroc"],
        "val_auprc": val_metrics_05["auprc"],
        "val_best_threshold": val_metrics_opt["threshold"],
        "val_f1_opt": val_metrics_opt["f1"],
        "val_mcc_opt": val_metrics_opt["mcc"],
    }
    history.append(row)

    print(
        f"Epoch {epoch:02d} | "
        f"Train Loss {train_metrics['loss']:.4f} AUC {train_metrics['auroc']:.4f} MCC {train_metrics['mcc']:.4f} | "
        f"Val Loss {val_metrics_05['loss']:.4f} AUC {val_metrics_05['auroc']:.4f} "
        f"MCC@0.5 {val_metrics_05['mcc']:.4f} MCC@opt {val_metrics_opt['mcc']:.4f} "
        f"(thr={val_metrics_opt['threshold']:.3f})"
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": fusionai_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {k: getattr(FusionAIBenchmarkConfig, k) for k in dir(FusionAIBenchmarkConfig) if k.isupper()},
            "val_metrics_05": val_metrics_05,
            "val_metrics_opt": val_metrics_opt,
        },
        last_ckpt_path
    )

    current_val_auc = val_metrics_05["auroc"]
    if current_val_auc > best_val_auc:
        best_val_auc = current_val_auc
        best_epoch = epoch
        best_val_threshold = val_metrics_opt["threshold"]
        epochs_without_improvement = 0

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": fusionai_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_auc": best_val_auc,
                "best_val_threshold": best_val_threshold,
                "config": {k: getattr(FusionAIBenchmarkConfig, k) for k in dir(FusionAIBenchmarkConfig) if k.isupper()},
                "val_metrics_05": val_metrics_05,
                "val_metrics_opt": val_metrics_opt,
            },
            best_ckpt_path
        )
        print(f"   ★ New best checkpoint saved (Val AUC={best_val_auc:.4f})")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= FusionAIBenchmarkConfig.EARLY_STOPPING_PATIENCE:
            print("   ⏹ Early stopping triggered.")
            break

history_df = pd.DataFrame(history)
history_df.to_csv(os.path.join(FusionAIBenchmarkConfig.OUTPUT_DIR, "fusionai_training_history.csv"), index=False)

print("\n✅ Training complete.")
print(f"   Best epoch: {best_epoch}")
print(f"   Best Val AUC: {best_val_auc:.4f}")
print(f"   Best Val Threshold (MCC): {best_val_threshold:.3f}")

# ==============================================================================
# CELL 11: INTERNAL BLIND TEST EVALUATION
# ==============================================================================
print("\n🧪 Evaluating best FusionAI model on INTERNAL blind test set...")

ckpt = torch.load(best_ckpt_path, map_location=device)
fusionai_model.load_state_dict(ckpt["model_state_dict"])
fusionai_model.to(device)
fusionai_model.eval()

test_metrics_05, test_metrics_opt_from_test_only, test_true_fai, test_probs_fai, test_keys_fai = evaluate_loader(
    fusionai_model,
    test_loader_fai,
    criterion,
    device,
    use_amp=FusionAIBenchmarkConfig.USE_AMP,
)

# Fair thresholded evaluation: use validation-selected threshold
test_metrics_valopt = compute_binary_metrics(
    test_true_fai,
    test_probs_fai,
    threshold=ckpt["best_val_threshold"]
)

print("\n🏆 INTERNAL BLIND TEST RESULTS")
print("Threshold = 0.50")
print(f"   Accuracy:  {test_metrics_05['acc']:.4f}")
print(f"   Precision: {test_metrics_05['prec']:.4f}")
print(f"   Recall:    {test_metrics_05['rec']:.4f}")
print(f"   F1 Score:  {test_metrics_05['f1']:.4f}")
print(f"   MCC:       {test_metrics_05['mcc']:.4f}")
print(f"   AUROC:     {test_metrics_05['auroc']:.4f}")
print(f"   AUPRC:     {test_metrics_05['auprc']:.4f}")

print("\nThreshold = best validation MCC threshold")
print(f"   Threshold: {ckpt['best_val_threshold']:.3f}")
print(f"   Accuracy:  {test_metrics_valopt['acc']:.4f}")
print(f"   Precision: {test_metrics_valopt['prec']:.4f}")
print(f"   Recall:    {test_metrics_valopt['rec']:.4f}")
print(f"   F1 Score:  {test_metrics_valopt['f1']:.4f}")
print(f"   MCC:       {test_metrics_valopt['mcc']:.4f}")
print(f"   AUROC:     {test_metrics_valopt['auroc']:.4f}")
print(f"   AUPRC:     {test_metrics_valopt['auprc']:.4f}")

# Save internal test predictions
internal_test_results = test_df_fai.copy()
internal_test_results["True_Label"] = test_true_fai
internal_test_results["FusionAI_Prob"] = test_probs_fai
internal_test_results["FusionAI_Pred_0.50"] = (test_probs_fai >= 0.5).astype(int)
internal_test_results["FusionAI_Pred_ValOpt"] = (test_probs_fai >= ckpt["best_val_threshold"]).astype(int)
internal_test_results["FusionAI_ValOpt_Threshold"] = ckpt["best_val_threshold"]

internal_test_results["Result_Type_0.50"] = np.select(
    [
        (internal_test_results["True_Label"] == 1) & (internal_test_results["FusionAI_Pred_0.50"] == 1),
        (internal_test_results["True_Label"] == 0) & (internal_test_results["FusionAI_Pred_0.50"] == 1),
        (internal_test_results["True_Label"] == 1) & (internal_test_results["FusionAI_Pred_0.50"] == 0),
        (internal_test_results["True_Label"] == 0) & (internal_test_results["FusionAI_Pred_0.50"] == 0),
    ],
    ["TP", "FP", "FN", "TN"],
    default="ERR"
)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
internal_csv_path = os.path.join(FusionAIBenchmarkConfig.OUTPUT_DIR, f"fusionai_internal_test_predictions_{timestamp}.csv")
internal_generic_path = os.path.join(FusionAIBenchmarkConfig.OUTPUT_DIR, "fusionai_internal_test_predictions.csv")
internal_test_results.to_csv(internal_csv_path, index=False)
internal_test_results.to_csv(internal_generic_path, index=False)

metrics_json = {
    "best_epoch": int(best_epoch),
    "best_val_auc": float(best_val_auc),
    "best_val_threshold": float(ckpt["best_val_threshold"]),
    "internal_test_metrics_05": {k: float(v) for k, v in test_metrics_05.items()},
    "internal_test_metrics_valopt": {k: float(v) for k, v in test_metrics_valopt.items()},
}
with open(os.path.join(FusionAIBenchmarkConfig.OUTPUT_DIR, "fusionai_internal_metrics.json"), "w") as f:
    json.dump(metrics_json, f, indent=2)

print(f"\n💾 Internal test predictions saved to:")
print(f"   {internal_csv_path}")
print(f"   {internal_generic_path}")

plot_test_results(test_true_fai, test_probs_fai, title_prefix="FusionAI Internal Blind Test")

# ==============================================================================
# CELL 12: EXTERNAL TEST_CSV EVALUATION (POSITIVE-ONLY)
# ==============================================================================
if ext_loader_fai is not None:
    print("\n🧪 Evaluating external TEST_CSV (positive-only)...")

    ext_probs_fai, ext_keys_fai = evaluate_positive_only(
        fusionai_model,
        ext_loader_fai,
        device,
        use_amp=FusionAIBenchmarkConfig.USE_AMP,
    )

    ext_results = ext_test_pos_df_fai.copy()
    ext_results["FusionAI_Prob"] = ext_probs_fai
    ext_results["FusionAI_Pred_0.50"] = (ext_probs_fai >= 0.5).astype(int)
    ext_results["FusionAI_Pred_ValOpt"] = (ext_probs_fai >= ckpt["best_val_threshold"]).astype(int)
    ext_results["FusionAI_ValOpt_Threshold"] = ckpt["best_val_threshold"]

    ext_csv_path = os.path.join(FusionAIBenchmarkConfig.OUTPUT_DIR, "fusionai_external_TEST_CSV_predictions.csv")
    ext_results.to_csv(ext_csv_path, index=False)

    print("📊 EXTERNAL TEST_CSV SUMMARY (positive-only)")
    print(f"   Samples: {len(ext_results):,}")
    print(f"   Mean predicted probability: {ext_results['FusionAI_Prob'].mean():.4f}")
    print(f"   Median predicted probability: {ext_results['FusionAI_Prob'].median():.4f}")
    print(f"   Positive rate @ 0.50: {(ext_results['FusionAI_Pred_0.50'].mean()):.4f}")
    print(f"   Positive rate @ val-opt threshold ({ckpt['best_val_threshold']:.3f}): {(ext_results['FusionAI_Pred_ValOpt'].mean()):.4f}")
    print(f"   Saved external predictions to: {ext_csv_path}")
else:
    print("\nℹ️ No external TEST_CSV evaluation performed because TEST_CSV was not found or was empty.")

# ==============================================================================
# CELL 13: OPTIONAL COMPARISON AGAINST FUSELENS RESULTS
# ==============================================================================
fuselens_results_path = os.path.join(Config.OUTPUT_DIR, "final_test_results_detailed.csv")

if os.path.exists(fuselens_results_path):
    print("\n🔗 FuseLens results detected. Building side-by-side comparison...")
    fuselens_df = pd.read_csv(fuselens_results_path)

    if "sample_key" not in fuselens_df.columns:
        if "sequence" in fuselens_df.columns:
            fuselens_df["sample_key"] = fuselens_df["sequence"].astype(str).apply(md5_text)
        else:
            print("⚠️ FuseLens results file found but it does not contain 'sample_key' or 'sequence'.")
            fuselens_df = None

    if fuselens_df is not None:
        merge_cols = ["sample_key"]
        if "True_Label" in fuselens_df.columns:
            merge_cols.append("True_Label")
        if "Fusion_Prob" in fuselens_df.columns:
            merge_cols.append("Fusion_Prob")
        if "Predicted_Label" in fuselens_df.columns:
            merge_cols.append("Predicted_Label")

        comparison_df = internal_test_results.merge(
            fuselens_df[merge_cols].drop_duplicates(subset=["sample_key"]),
            on="sample_key",
            how="inner",
            suffixes=("_FusionAI", "_FuseLens")
        )

        if len(comparison_df) == 0:
            print("⚠️ Could not align FusionAI and FuseLens predictions by sample_key.")
        else:
            if "True_Label" in comparison_df.columns:
                y_true_compare = comparison_df["True_Label"].astype(int).to_numpy()
            else:
                y_true_compare = comparison_df["label"].astype(int).to_numpy()

            fusionai_metrics_compare = compute_binary_metrics(
                y_true_compare,
                comparison_df["FusionAI_Prob"].to_numpy(),
                threshold=0.5
            )
            fuselens_metrics_compare = compute_binary_metrics(
                y_true_compare,
                comparison_df["Fusion_Prob"].to_numpy(),
                threshold=0.5
            )

            summary_df = pd.DataFrame([
                {
                    "Model": "FuseLens",
                    "Accuracy": fuselens_metrics_compare["acc"],
                    "Precision": fuselens_metrics_compare["prec"],
                    "Recall": fuselens_metrics_compare["rec"],
                    "F1": fuselens_metrics_compare["f1"],
                    "MCC": fuselens_metrics_compare["mcc"],
                    "AUROC": fuselens_metrics_compare["auroc"],
                    "AUPRC": fuselens_metrics_compare["auprc"],
                },
                {
                    "Model": "FusionAI",
                    "Accuracy": fusionai_metrics_compare["acc"],
                    "Precision": fusionai_metrics_compare["prec"],
                    "Recall": fusionai_metrics_compare["rec"],
                    "F1": fusionai_metrics_compare["f1"],
                    "MCC": fusionai_metrics_compare["mcc"],
                    "AUROC": fusionai_metrics_compare["auroc"],
                    "AUPRC": fusionai_metrics_compare["auprc"],
                },
            ])

            print("\n📊 SIDE-BY-SIDE COMPARISON ON ALIGNED INTERNAL TEST SAMPLES")
            print(summary_df.to_string(index=False))

            comparison_save_path = os.path.join(FusionAIBenchmarkConfig.OUTPUT_DIR, "fuselens_vs_fusionai_comparison.csv")
            aligned_preds_path = os.path.join(FusionAIBenchmarkConfig.OUTPUT_DIR, "fuselens_vs_fusionai_aligned_predictions.csv")

            summary_df.to_csv(comparison_save_path, index=False)
            comparison_df.to_csv(aligned_preds_path, index=False)

            print(f"\n💾 Comparison saved to:")
            print(f"   {comparison_save_path}")
            print(f"   {aligned_preds_path}")
else:
    print("\nℹ️ FuseLens final_test_results_detailed.csv not found. Skipping direct comparison.")

print("\n✅ FusionAI raw-file benchmark pipeline complete.")
