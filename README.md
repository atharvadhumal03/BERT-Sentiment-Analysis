# 💬 BERT Sentiment Analysis on Amazon Reviews

Fine-tuning `bert-base-uncased` on 3.6 million Amazon product reviews for binary sentiment classification, trained on Northeastern University's Explorer HPC cluster. Achieves a **F1 score of 97.06%** and **Accuracy of 97.06%** on the held-out test set.

---

## 📊 Results

| Metric | Score |
|--------|-------|
| F1 (macro) | **97.06%** |
| Accuracy | **97.06%** |
| Test set size | 400,000 reviews |

F1 improved from **96.5% → 97.06%** across 3 epochs. Most errors occur on short or ambiguous reviews — see the [error analysis notebook](notebooks/error_analysis.ipynb) for a full breakdown.

**Links:** [HuggingFace Hub](https://huggingface.co/atharvadhumal/bert-amazon-polarity) · [W&B Training Curves](https://wandb.ai/atharva-dhumal07-student/bert-sentiment-amazon)

---

## 📁 Project Structure

```
BERT-Sentiment-Analysis/
├── src/
│   ├── train.py              # Fine-tuning with HuggingFace Trainer
│   ├── preprocess.py         # Tokenization and dataset caching
│   └── inference.py          # Batch inference with confidence scores
├── scripts/
│   └── train.sh              # SLURM job script for HPC
├── notebooks/
│   └── error_analysis.ipynb  # Misclassification analysis with outputs
├── data/tokenized/           # Cached tokenized dataset (gitignored)
└── output/                   # Model checkpoints (gitignored)
```

---

## 📦 Dataset

- **Source:** [Amazon Polarity](https://huggingface.co/datasets/fancyzhx/amazon_polarity) via HuggingFace Datasets
- **Content:** Amazon product reviews labelled positive (1) or negative (0)
- **Size:** 3,600,000 train / 400,000 test
- **Preprocessing:** Tokenized with `bert-base-uncased` tokenizer (max length 512), cached to disk to avoid re-processing on each run

---

## ⚙️ Setup

**Clone the repository:**
```bash
git clone https://github.com/atharvadhumal03/BERT-Sentiment-Analysis
cd BERT-Sentiment-Analysis
```

**Create conda environment:**
```bash
conda env create -f environment.yml
conda activate bert-sentiment
```

---

## 🚀 Inference

**Run on any text from the command line:**
```bash
python src/inference.py \
  --text "Absolutely love this, works perfectly!" \
          "Stopped working after two days. Total waste of money."
```

```
[POSITIVE] (99.8%)  Absolutely love this, works perfectly!
[NEGATIVE] (99.7%)  Stopped working after two days. Total waste of money.
```

**Or use the HuggingFace pipeline directly:**
```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="atharvadhumal/bert-amazon-polarity")
classifier("This product is absolutely amazing!")
# [{'label': 'LABEL_1', 'score': 0.998}]
```

**Or use as a library:**
```python
from src.inference import load_model, predict

tokenizer, model, device = load_model("atharvadhumal/bert-amazon-polarity")
results = predict(["Great product!", "Awful experience."], tokenizer, model, device)
# [{'label': 'positive', 'confidence': 0.998}, {'label': 'negative', 'confidence': 0.997}]
```

---

## 🏋️ Training

**Preprocess and cache the dataset:**
```bash
python src/preprocess.py
```

**Train locally:**
```bash
python src/train.py
```

**Submit to HPC (SLURM):**
```bash
sbatch scripts/train.sh
```

---

## 🧠 Implementation Details

| Component | Details |
|-----------|---------|
| Framework | PyTorch + HuggingFace Transformers |
| Base Model | `bert-base-uncased` (110M parameters) |
| Loss Function | Cross-Entropy |
| Optimizer | AdamW |
| Learning Rate | 2e-5 with linear warmup (6%) |
| Weight Decay | 0.01 |
| Batch Size | 64 |
| Epochs | 3 |
| Precision | fp16 |
| Training Hardware | Tesla V100-SXM2-32GB (ep 1–2) · NVIDIA H200 (ep 3) |

---

## 💡 Key Learnings

- **Disk quota management** — saving a checkpoint every 1,000 steps across 168k total steps consumed ~34GB; switching to `save_steps=5000` and `save_total_limit=2` kept disk usage under control
- **HPC network isolation** — compute nodes had no internet access, requiring `local_files_only=True` for all HuggingFace model loads and `WANDB_MODE=offline` for experiment tracking
- **Checkpoint resuming** — training was split across multiple jobs due to 8-hour wall limits; the `RESUME_FROM_CHECKPOINT` env var allowed seamless resumption with no loss of training state
- **W&B offline sync** — metrics were logged locally during training and synced to the dashboard afterwards with `wandb sync`

---

## 📋 Requirements

See `environment.yml` for full dependencies. Key packages:
- `transformers`
- `datasets`
- `torch`
- `scikit-learn`
- `wandb`
