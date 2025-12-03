# FuseLens: Long-Context Genomic Foundation Models for DNA Breakpoint Detection

**FuseLens** is a deep learning framework designed to detect and validate gene fusion breakpoints with high precision. By leveraging **HyenaDNA**—a genomic foundation model capable of processing long context windows (20kb)—FuseLens overcomes the limitations of traditional short-read alignment tools in identifying complex or repetitive fusion events.

---

## Key Features

* **Foundation Model Backbone:** Uses `hyenadna-small-32k` to process 20kb genomic windows at single-nucleotide resolution.
* **Attention Pooling Head:** A custom learnable pooling layer that localizes the exact breakpoint signal within the massive 20kb context.
* **Hybrid Clinical Pipeline:** Integrates with **CTAT-LR-fusion** (Nanopore/PacBio) to act as a high-precision validation layer for long-read discovery.
* **Robust Data Engineering:** Implements a mathematically rigorous "Hard Negative" strategy to eliminate false positives.

---

## Methodology

### 1. Model Architecture
Instead of standard CNNs or quadratic Transformers, we utilized **HyenaDNA**, which uses implicit long convolutions to scale sub-quadratically ($O(N \log N)$).
* **Input:** 20,480 bp genomic sequence (centered on candidate breakpoint).
* **Backbone:** Pretrained Hyena operators (feature extraction).
* **Head:** **Attention Pooling**. Instead of max-pooling, the model calculates an attention score $\alpha_t$ for every nucleotide, allowing it to "focus" on the junction and ignore flanking noise.
* **Localization:** The model outputs both a binary classification probability and a specific **Breakpoint Index** derived from the peak attention weight.

### 2. Data Engineering (Hard-Negative Strategy)
To prevent the model from learning simple shortcuts (like GC-content bias), we engineered a robust training dataset ($N \approx 27,000$):
* **Canonical Baseline:** 5,000 real human transcripts (UniProt Swiss-Prot) labeled as Negative (0).
* **Synthetic Hard Negatives:**
    * **Reversed ($\mathcal{R}$):** Tests directionality.
    * **Shuffled ($\mathcal{P}_{rand}$):** Preserves GC-content but destroys syntax (Tests motif learning).
    * **Random Pairs ($\mathcal{J}$):** Randomly joined genes (Tests structural validity).
* **Domain Adaptation:** We used **Regularized Empirical Risk Minimization (R-ERM)** by mixing synthetic data with real RNA-Seq reads to bridge the "sim-to-real" gap.

---

## The Pipeline

### Phase 1: Discovery (Long-Read)
We use **CTAT-LR-fusion** to parse raw Oxford Nanopore (ONT) or PacBio FASTQ files.
* **Role:** Candidate Generator.
* **Method:** Uses `minimap2` to find reads mapping to two different genes.
* **Output:** List of candidate fusions with *approximate* coordinates.

### Phase 2: Bridging (Context Extraction)
A custom Python script takes CTAT candidates and fetches the clean **$\pm$10kb genomic context** from the reference genome (`hg38`).
* **Why?** Nanopore reads have high error rates (5-10%). Extracting the reference context ensures the model sees clean motifs while verifying the structural arrangement found by the long read.

### Phase 3: Validation (FuseLens)
The extracted sequences are fed into the fine-tuned FuseLens model.
* **Classification:** Assigns a probability score ($P > 0.9$ = High Confidence).
* **Refinement:** The attention mechanism corrects the approximate CTAT coordinate to the precise biological breakpoint.

---

## Performance Benchmark

Evaluated on a hold-out test set of 21,952 samples:

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **AUC** | **0.9805** | Near-perfect discrimination capability. |
| **Recall** | **95.88%** | Captures ~96% of true fusions (Crucial for diagnosis). |
| **Precision** | **94.83%** | Very low False Alarm rate (5.2%). |
| **F1 Score** | **0.9535** | Excellent balance between sensitivity and specificity. |

---

## Repository Structure

* `HyenaDNA_Breakpoint_Classifier.ipynb`: **Training Pipeline.** Data loading, augmentation, model fine-tuning, and evaluation.
* `FuseLens_Inference_Pipeline.ipynb`: **Deployment Pipeline.** End-to-end workflow from FASTQ $\to$ Clinical Report.
* `data_pipeline.py`: Scripts for generating synthetic hard negatives from UniProt.
* `hyenadna_breakpoint_model/`: Directory containing the saved model weights (`model.safetensors`) and tokenizer.

---

## Future Directions
* **Variant Effect Prediction:** Repurposing the backbone to score SNP pathogenicity.
* **Multi-Class Segmentation:** Upgrading the model to distinguish between *cis-splicing* (benign) and *translocations* (cancerous).
