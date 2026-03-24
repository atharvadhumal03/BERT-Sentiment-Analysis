import os
import sys
import torch
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import preprocess


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = float((preds == labels).mean())
    return {"accuracy": acc}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    cache_dir = os.environ.get("DATA_CACHE_DIR", "./data/tokenized")
    print(f"Loading dataset (cache: {cache_dir})...")
    dataset = preprocess(cache_dir=cache_dir)

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )

    output_dir = os.environ.get("OUTPUT_DIR", "./output")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=128,
        learning_rate=2e-5,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=100,
        dataloader_num_workers=4,
        # Uncomment to cap training for quick experiments:
        # max_steps=5000,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(os.path.join(output_dir, "best_model"))
    print(f"Training complete. Model saved to {output_dir}/best_model")


if __name__ == "__main__":
    main()
