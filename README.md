# FuseLens: Attention-Guided Gene Fusion Classification via Long-Context Genomic Foundation Models

**FuseLens** is a deep learning framework designed to detect and validate gene fusion breakpoints with high discriminative precision. By fine-tuning **HyenaDNA**—a genomic foundation model capable of processing sub-quadratic long context windows—FuseLens overcomes the limitations of traditional short-read alignment tools in identifying complex or repetitive fusion events.

---

## Key Features

* **Foundation Model Backbone:** Fine-tunes `hyenadna-small-32k` to process **32kb genomic windows** (32,768 bp) at single-position resolution.
* **Weakly Supervised Localization:** A custom learnable attention head that acts as a "Soft Pointer," identifying the genomic neighborhood (median width ~2kb) of the fusion event without requiring dense, nucleotide-level segmentation labels.
* **Shift-Invariant Architecture:** Utilizes a **Random Jitter** operator during training to enforce translation invariance, preventing the model from relying on positional heuristics (e.g., center bias).
* **Hard-Negative Engineering:** Implements a mathematically rigorous negative space using **Ensembl CDS**, Reverse Complements, and Circular Permutations to eliminate "easy" discriminative shortcuts.

---

## Methodology

### 1. Model Architecture
Instead of standard CNNs or quadratic Transformers, we utilize HyenaDNA, which employs implicit long convolutions to scale sub-quadratically at $O(N \log N)$.

* **Input:** 32,768 bp genomic sequence (centered on candidate breakpoint).
* **Backbone:** Pretrained Hyena operators (feature extraction).
* **Head:** **Attention Pooling**. The model calculates an attention score $\alpha_t$ for every nucleotide, aggregating evidence across the gene scale.
* **Output:** A fusion probability score $\hat{p}$ and a **Localization Proxy** $\hat{\ell}$ (derived from peak attention), reported only under high confidence ($\tau=0.9$).

### 2. Data Engineering (Hard-Negative Strategy)
To prevent the model from learning simple shortcuts (like GC-content bias or protein-coding syntax), we engineered a robust dataset ($N = 253,925$):

* **Canonical Baseline:** Real human **Coding DNA Sequences (CDS)** from Ensembl (mapped from UniProt) labeled as Negative (0).
* **Synthetic Hard Negatives:**
    * **Reverse Complement ($\mathcal{RC}$):** Enforces strand awareness ($5' \to 3'$ syntax).
    * **Grammar Inversion ($\mathcal{R}_{seq}$):** Reverses sequence without complementation to break grammatical structure while preserving local stats.
    * **Circular Permutation ($\mathcal{J}$):** Creates "decoy junctions" by rotating a single gene, forcing the model to distinguish true fusions from structural discontinuities.

---

## The Pipeline

### Phase 1: Candidate Generation
FuseLens is designed as a post-caller validation layer. It accepts candidates from:
* Standard RNA-seq callers (STAR-Fusion, Arriba).
* Long-read structural variant callers (CTAT-LR-fusion).

### Phase 2: Context Extraction & Jitter
A custom pre-processor fetches the clean **$\pm$16kb genomic context** from the reference genome (`hg38`) around the candidate junction.
* **Random Jitter:** The core signal is uniformly placed within the 32kb window (via random DNA padding or cropping) to ensure the model scans the entire context.

### Phase 3: Inference (FuseLens)
* **Classification:** Assigns a probability score.
* **Localization:** If $P > 0.9$, the attention mechanism highlights the specific region contributing to the fusion classification, filtering out artifacts and alignment noise.

---

## Performance Benchmark

Evaluated on a stratified blind test set of **38,089 samples**:

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **AUC** | **0.9794** | Exceptional ranking capability; clearly separates fusions from hard decoys. |
| **Accuracy** | **93.07%** | High reliability across balanced positive and negative classes. |
| **Precision** | **90.83%** | Minimizes False Positives, a critical requirement for clinical pipelines. |
| **Recall** | **90.40%** | Captures the vast majority of true pathogenic events. |

---

## Repository Structure

* `FuseLens_Training_Pipeline.ipynb`: **Training Pipeline.** Data loading, random jitter augmentation, model fine-tuning, and validation loops.
* `FuseLens_Inference_Pipeline.ipynb`: **Deployment Pipeline.** End-to-end workflow from Candidate List $\to$ Validated Report with Attention Plots.
* `FuseLens_Negative_Data_Pipeline.ipynb`: Scripts for generating synthetic hard negatives using **Ensembl CDS** (fixing UniProt protein vs. DNA issues).
* `hyenadna_breakpoint_model/`: Directory containing the saved model weights (`model.safetensors`) and tokenizer.

---

## Future Directions

* **Synergy with Nanopore:** Native processing of full-length reads from Oxford Nanopore Technologies (ONT) to validate isoforms without assembly.
* **Multi-Class Segmentation:** Upgrading the model to distinguish between *cis-splicing* (benign read-throughs) and *translocations* (cancerous fusions).
