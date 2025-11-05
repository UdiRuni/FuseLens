# FuseLens - The Fusion Transcripts Booster 

### Goal: 
Develop a deep learning system for high-accuracy prioritization of cancer-associated fusion transcripts by interpreting their underlying sequence "language." 
### Key Features: 
Fine-tune a pre-trained DNN model – DeepVariant (https://github.com/google/deepvariant) to classify the nucleotide sequences surrounding a fusion breakpoints as images. 
Use the model as classfier for new fusion transcripts false/true-postive predication with higher accuracy and compare with other tools (i.e.Arriba, FusionScan, and GFusion, which are algorithms designed to analyze RNA sequencing data to identify split reads and other evidence of fusion transcripts).
### Impact: 
Accelerated Science Discovery -  automates the prioritization of high-impact driver fusions from thousands of candidates, speeding up the discovery of novel cancer biomarkers and therapeutic targets. 
### Keywords: 
Fusion Transcript, Chimeric RNA, DNN, Genomic Language Model, Computer Graphics, GenAI, Sequence Classification, Attention Mechanism, Bioinformatics, Cancer Genomics.
### Data Sources: 
The Cancer Genome Atlas (TCGA): Source for tumor RNA-Seq and matched Whole-Genome Sequencing (WGS) data to generate ground truth labels.
COSMIC & ChimerDB: A "gold-standard" set of positive examples.
GTEx Portal: Source of normal tissue RNA-Seq data to identify and filter out benign read-through events.
### Expected challenges: 
Data Scarcity & Quality: Obtaining a large, high-quality labeled dataset is a bottleneck. Creating ground truth labels is a non-trivial task.
Computational Cost: Fine-tuning large models requires significant GPU resources and time.
Model Interpretability: Translating scores into verifiable hypotheses is a complex and active area of research.
Generalization: The model may not generalize well across different cancer types or sequencing technologies (e.g., short-read vs. long-read) without specific training strategies.
### Tentative methods:
Data Curation: Generate labels (True/False Positives) by comparing RNA-Seq fusion calls with matched WGS data from TCGA
Input Preparation: For each fusion, extract the raw DNA sequence surrounding the breakpoint from a reference genome (e.g., using pyfaidx).
Model Fine-Tuning: Load a pre-trained genomic Model and fine-tune it for sequence classification using the labeled breakpoint sequences.
Interpretation: Analyze the model's attention scores to identify predictive nucleotide patterns using tools like bertviz.

<img width="1967" height="1069" alt="image" src="https://github.com/user-attachments/assets/7b912c35-5386-4c0c-979f-8d6874716fbe" />

