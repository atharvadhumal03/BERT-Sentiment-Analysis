import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS = {0: "negative", 1: "positive"}


def load_model(model_path: str):
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def predict(texts: list[str], tokenizer, model, device: str, batch_size: int = 32) -> list[dict]:
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu()
        for j, text in enumerate(batch):
            pred = int(probs[j].argmax())
            results.append({
                "text": text,
                "label": LABELS[pred],
                "confidence": round(float(probs[j][pred]), 4),
            })
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BERT Sentiment Inference")
    parser.add_argument("--model", default=os.environ.get("MODEL_PATH", "atharvadhumal/bert-amazon-polarity"))
    parser.add_argument("--text", nargs="+", help="One or more review texts to classify")
    args = parser.parse_args()

    tokenizer, model, device = load_model(args.model)
    print(f"Model loaded on {device}\n")

    texts = args.text or [
        "This product is absolutely amazing, I love it!",
        "Terrible quality, broke after one day. Complete waste of money.",
        "It's okay, nothing special but gets the job done.",
    ]

    results = predict(texts, tokenizer, model, device)
    for r in results:
        print(f"[{r['label'].upper():8s}] ({r['confidence']:.1%})  {r['text'][:80]}")
