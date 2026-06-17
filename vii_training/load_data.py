from datasets import load_dataset

def load_training_data(max_samples: int = None):
    """Download wikitext-103 - clean wikipedia text"""
    print(f'loading dataset: wikitext-103-raw-v1...')
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
    texts = [item["text"] for item in dataset if item["text"].strip()]
    if max_samples:
        texts = texts[:max_samples]
    print(f"Loaded {len(texts):,} documents")
    return texts