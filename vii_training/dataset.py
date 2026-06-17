import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    """
    Prepares text data by splitting into training chunks.
    The model learns to predict the next token. Each chunk provides
    input-target pairs for next-token prediction

    Each sample: input[t] and target[t+1] for all positions t
    this is called 'teacher forcing' - we show the correct
    answer for every position during training
    """

    def __init__(self, texts: list[str], tokenizer, max_seq_len: int = 1024):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # concatenate all texts with EOS seperators
        # EOS prevents the model from learning false connections
        # between unrelated documents
        all_tokens = []
        for text in texts:
            tokens = tokenizer.encode(text)
            all_tokens.append(tokens)
            all_tokens.append(tokenizer.eos_token_id) # document boundary marker

        self.tokens = torch.tensor(all_tokens, dtype=torch.long)
        print(f'Total tokens in dataset: {len(self.tokens)}')

    def __len__(self) -> int:
        """
        Number of chunks. each uses max_seq_len + 1 tokens
        """
        return (len(self.tokens) - 1) // self.max_seq_len
    
    def __getitem__(self, idx: int) -> tuple:
        """
        Returns (input_ids, target_ids) for one chunk
        target is shifted by 1 position.

        tokens: [the, cat, sat, on, the, mat, EOS, the, dog, ...]
        idx=0:  [the, cat, sat, on, the ] <- input_ids
                [cat, sat, on, the, mat] <- target_ids (shifted)
        """
        start = idx * self.max_seq_len
        end = start + self.max_seq_len
        input_ids = self.tokens[start: end]
        target_ids = self.tokens[start+1: end+1]
        return input_ids, target_ids