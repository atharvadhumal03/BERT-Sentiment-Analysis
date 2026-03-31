import os
import sys
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
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
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro"),
    }


def main():
    data_cache_dir = os.environ.get("DATA_CACHE_DIR", "./data/tokenized")
    output_dir = os.environ.get("OUTPUT_DIR", "./output")
    wandb_run_name = os.environ.get("WANDB_RUN_NAME", None)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading dataset (cache: {data_cache_dir})...")
    dataset = preprocess(cache_dir=data_cache_dir)
    print(f"Train: {len(dataset['train'])} | Test: {len(dataset['test'])}")

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=128,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.06,
        fp16=(device == "cuda"),
        dataloader_num_workers=4,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        logging_steps=100,
        report_to="wandb",
        run_name=wandb_run_name,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("\nFinal evaluation:")
    results = trainer.evaluate()
    print(f"  loss:     {results['eval_loss']:.4f}")
    print(f"  accuracy: {results['eval_accuracy']:.4f}")
    print(f"  f1:       {results['eval_f1']:.4f}")

    trainer.save_model(os.path.join(output_dir, "best_model"))
    print(f"\nTraining complete. Model saved to {os.path.join(output_dir, 'best_model')}")


if __name__ == "__main__":
    main()
