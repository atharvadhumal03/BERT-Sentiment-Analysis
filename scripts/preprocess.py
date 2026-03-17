from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    texts = [t + " " + c for t, c in zip(batch['title'], batch['content'])]
    return tokenizer(texts, max_length=128, truncation=True, padding='max_length')

def preprocess():
    dataset = load_dataset("amazon_polarity")
    print("Starting tokenization...")
    tokenized_dataset = dataset.map(tokenize, batched=True)
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset = tokenized_dataset.remove_columns(["title", "content"])
    tokenized_dataset.set_format("torch")
    print("Done!")
    return tokenized_dataset