# HyenaDNA Binary Classifier for DNA Breakpoint Detection

This project fine-tunes HyenaDNA to classify DNA sequences as positive or negative breakpoints, similar to FusionAI.

## Overview

The pipeline includes:
1. **Data Preparation**: Merges and shuffles positive/negative CSV files
2. **Model Architecture**: Adds a binary classification head to HyenaDNA
3. **Fine-tuning**: Trains on 20kb DNA sequences
4. **Prediction**: Classifies test sequences and outputs probabilities

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Note: If you have GPU, ensure you have the right CUDA version for PyTorch
# Visit https://pytorch.org for installation instructions
```

## Data Format

Your CSV files should have the following columns (tab-separated):
- `Hgene`: Head gene name
- `Hchr`: Head chromosome
- `Hbp`: Head breakpoint position
- `Hstrand`: Head strand
- `Tgene`: Tail gene name
- `Tchr`: Tail chromosome
- `Tbp`: Tail breakpoint position
- `Tstrand`: Tail strand
- `5'-gene sequence (10Kb)`: 5' sequence (10,000 bp)
- `3'-gene sequence (10Kb)`: 3' sequence (10,000 bp)

The script will automatically merge the 5' and 3' sequences to create 20kb input sequences.

## Quick Start

### 1. Prepare Your Data Files

Place your data files in the same directory or update the paths in the script:
- `positive_breakpoints.csv` - Positive breakpoint examples
- `negative_breakpoints.csv` - Negative breakpoint examples
- `test_breakpoints.csv` - Test data for predictions

### 2. Update Configuration

Edit the `main()` function in `hyenadna_breakpoint_classifier.py`:

```python
# Update these paths
POSITIVE_CSV = "path/to/your/positive_breakpoints.csv"
NEGATIVE_CSV = "path/to/your/negative_breakpoints.csv"
TEST_CSV = "path/to/your/test_breakpoints.csv"
```

### 3. Choose Model Size

Select a HyenaDNA model based on your computational resources:

```python
# Recommended for 20kb sequences with moderate GPU (RTX 3090, A100):
MODEL_NAME = "LongSafari/hyenadna-small-32k-seqlen"

# For smaller GPUs or faster training:
MODEL_NAME = "LongSafari/hyenadna-tiny-1k-seqlen"

# For more powerful systems:
MODEL_NAME = "LongSafari/hyenadna-medium-160k-seqlen"
```

### 4. Run Training

```bash
python hyenadna_breakpoint_classifier.py
```

### 5. View Results

After training completes, you'll find:
- `hyenadna_breakpoint_model/final_model/` - Trained model weights
- `breakpoint_predictions.csv` - Test predictions with probabilities

## Training Parameters

Adjust these in the `main()` function based on your needs:

```python
trainer = pipeline.train(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    num_epochs=3,           # Number of training epochs
    batch_size=4,           # Reduce if out of memory
    learning_rate=2e-5,     # Learning rate
    warmup_steps=500,       # Warmup steps
    weight_decay=0.01       # Weight decay for regularization
)
```

### GPU Memory Guidelines

- **RTX 3090 (24GB)**: batch_size=4-8 with small model
- **A100 (40GB)**: batch_size=8-16 with small/medium model
- **A100 (80GB)**: batch_size=16-32 with medium/large model
- **CPU only**: batch_size=1-2 (very slow)

## Output Format

The `breakpoint_predictions.csv` file contains:
- All original columns from test CSV
- `predicted_label`: 0 (negative) or 1 (positive)
- `breakpoint_probability`: Probability of being a positive breakpoint (0-1)
- `prediction`: Human-readable label ("Positive" or "Negative")

## Advanced Usage

### Using Only Specific Steps

You can run individual steps by modifying the script:

```python
# Only prepare data
data_prep = DataPreparator(POSITIVE_CSV, NEGATIVE_CSV)
train_df = data_prep.load_and_prepare_data()
train_df.to_csv("prepared_data.csv", index=False)

# Only make predictions with existing model
pipeline = HyenaDNATrainingPipeline(MODEL_NAME, OUTPUT_DIR)
predictions = pipeline.predict(
    test_csv_path=TEST_CSV,
    model_path="./hyenadna_breakpoint_model/final_model",
    output_path="new_predictions.csv"
)
```

### Evaluating on Test Data with Labels

If your test data has labels, modify the prediction function:

```python
# In the predict method, if you have true labels:
test_df['true_label'] = test_df['label']  # Use actual labels

# Calculate metrics
from sklearn.metrics import classification_report, confusion_matrix

y_true = test_df['true_label']
y_pred = test_df['predicted_label']

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
```

## Model Architecture Details

The HyenaDNA classifier consists of:

1. **Pretrained HyenaDNA Backbone**: Extracts features from 20kb DNA sequences
2. **Global Average Pooling**: Aggregates sequence representations
3. **Classification Head**:
   - Linear layer (hidden_size → 512) + ReLU + Dropout
   - Linear layer (512 → 128) + ReLU + Dropout
   - Linear layer (128 → 2) for binary classification

## Troubleshooting

### Out of Memory Error

- Reduce `batch_size` to 2 or 1
- Use a smaller model (tiny instead of small)
- Enable gradient checkpointing (add to TrainingArguments):
  ```python
  gradient_checkpointing=True
  ```

### Slow Training

- Ensure you're using GPU: check with `torch.cuda.is_available()`
- Reduce `num_epochs` for initial testing
- Use a smaller model for prototyping

### Poor Performance

- Increase training epochs (try 5-10)
- Adjust learning rate (try 1e-5 or 5e-5)
- Check data quality and balance
- Try data augmentation (reverse complement)

### Import Errors

Make sure all dependencies are installed:
```bash
pip install --upgrade -r requirements.txt
```

## Performance Expectations

With proper training, you should expect:
- **Accuracy**: 85-95% (depends on data quality)
- **Training time**: 
  - Small model, 1000 samples: ~30-60 minutes on A100
  - Medium model, 10000 samples: ~4-8 hours on A100

## Citation

If you use this code, please cite:

```bibtex
@article{nguyen2023hyenadna,
  title={HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution},
  author={Nguyen, Eric and Poli, Michael and Faizi, Marjan and Thomas, Armin and Birch-Sykes, Callum and Wornow, Michael and Patel, Aman and Rabideau, Clayton and Massaroli, Stefano and Bengio, Yoshua and Ermon, Stefano and Baccus, Stephen A. and R{\'e}, Christopher},
  journal={NeurIPS},
  year={2023}
}
```

## License

This code is provided for research purposes. Check HyenaDNA's license for model usage terms.

## Support

For issues:
1. Check this README
2. Review HyenaDNA documentation: https://github.com/HazyResearch/hyena-dna
3. Check Hugging Face model pages: https://huggingface.co/LongSafari
