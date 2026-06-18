from dataclasses import dataclass
import tiktoken

@dataclass
class TokenizerConfig:
    name: str = 'gpt2'
    vocab_size: int = 50257

class SimpleTokenizer:
    def __init__(self, config=None):
        self.config = config or TokenizerConfig()
        self.enc = tiktoken.get_encoding(self.config.name)
        self.eos_token = "<|endoftext|>"
        self.eos_token_id = self.enc.encode(
            self.eos_token, allowed_special={self.eos_token}
        )[0]

    def encode(self, text):
        return self.enc.encode(text, allowed_special={self.eos_token})

    def decode(self, ids):
        return self.enc.decode(ids)

    @property
    def vocab_size(self):
        return self.config.vocab_size