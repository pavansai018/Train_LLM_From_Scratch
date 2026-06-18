from datasets import load_dataset
from torch.utils.data import Dataset
import torch

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_seq_len=128):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        all_tokens = []
        for text in texts:
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)
            all_tokens.append(tokenizer.eos_token_id)
        self.tokens = torch.tensor(all_tokens, dtype=torch.long)

    def __len__(self):
        return (len(self.tokens) - 1) // self.max_seq_len

    def __getitem__(self, idx):
        start = idx * self.max_seq_len
        end = start + self.max_seq_len
        return self.tokens[start:end], self.tokens[start + 1 : end + 1]
    

def load_training_data(max_samples=None):
    print("Loading dataset: wikitext-103-raw-v1...")
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    texts = [item["text"] for item in dataset if item["text"].strip()]
    if max_samples:
        texts = texts[:max_samples]
    print(f"Loaded {len(texts):,} documents")
    return texts