# BERT Sentiment Analysis

Fine-tuning `bert-base-uncased` for binary sentiment classification on the [Amazon Polarity](https://huggingface.co/datasets/fancyzhx/amazon_polarity) dataset (3.6M training reviews).

## Results

| Metric | Score |
|--------|-------|
| **F1 (macro)** | **97.06%** |
| **Accuracy** | **97.06%** |
| Test samples | 400,000 |

Training curves available on [Weights & Biases](https://wandb.ai/atharva-dhumal07-student/bert-sentiment-amazon).

## Setup

```bash
conda env create -f environment.yml
conda activate bert-sentiment
```

## Usage

### Inference

```bash
python src/inference.py --model ./output/run_5545887/best_model \
  --text "This product is absolutely amazing!" \
           "Terrible quality, broke after one day."
```

Output:
```
[POSITIVE] (99.8%)  This product is absolutely amazing!
[NEGATIVE] (99.6%)  Terrible quality, broke after one day.
```

Or use it as a library:

```python
from src.inference import load_model, predict

tokenizer, model, device = load_model("./output/run_5545887/best_model")
results = predict(["Great product!", "Awful experience."], tokenizer, model, device)
```

### Training

**Preprocess** (tokenize and cache dataset):
```bash
python src/preprocess.py
```

**Train locally:**
```bash
python src/train.py
```

**Train on HPC (SLURM):**
```bash
sbatch scripts/train.sh
```

## Training Details

| Hyperparameter | Value |
|----------------|-------|
| Base model | `bert-base-uncased` |
| Epochs | 3 |
| Batch size | 64 |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Warmup ratio | 0.06 |
| Precision | fp16 |

Training was run on a **Tesla V100-SXM2-32GB** (epochs 1–2) and **NVIDIA H200** (epoch 3) on Northeastern University's Explorer HPC cluster.

## Project Structure

```
├── src/
│   ├── train.py          # Fine-tuning with HuggingFace Trainer
│   ├── preprocess.py     # Tokenization and dataset caching
│   └── inference.py      # Inference script
├── scripts/
│   └── train.sh          # SLURM job script
├── notebooks/            # Exploratory analysis
├── data/tokenized/       # Cached tokenized dataset (gitignored)
└── output/               # Model checkpoints (gitignored)
```
