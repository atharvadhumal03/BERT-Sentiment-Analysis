import os
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    texts = [t + " " + c for t, c in zip(batch['title'], batch['content'])]
    return tokenizer(texts, max_length=128, truncation=True, padding='max_length')

def preprocess(cache_dir="./data/tokenized"):
    if os.path.exists(cache_dir):
        print(f"Loading tokenized dataset from cache: {cache_dir}")
        tokenized_dataset = load_from_disk(cache_dir)
        tokenized_dataset.set_format("torch")
        return tokenized_dataset

    dataset = load_dataset("amazon_polarity")
    print("Starting tokenization...")
    tokenized_dataset = dataset.map(tokenize, batched=True, num_proc=4)
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset = tokenized_dataset.remove_columns(["title", "content"])
    tokenized_dataset.save_to_disk(cache_dir)
    print(f"Saved tokenized dataset to {cache_dir}")
    tokenized_dataset.set_format("torch")
    print("Done!")
    return tokenized_dataset
