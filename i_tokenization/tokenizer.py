from dataclasses import dataclass
import tiktoken
from typing import List

@dataclass
class TokenizerConfig:
    """
    GPT-2's pretrained BPE tokenizer.
    same Byte Pair Encoder as GPT-3/4 - 50K merges,
    battle tested on billions of documents and already traiend.

    vocab_size=50257. is the exact GPT-2 vocabulary size (50000 merges + 256 byte tokens + 1 EOS)
    big enough for rare subwords and small enough for fast matrix operations.
    """

    name: str = 'gpt2'
    vocab_size: int = 50257

class SimpleTokenizer:
    """
    Wraps tiktoken to give us a friendly, consistent interface.
    tiktoken's raw API is low-level (you need to add allowed_special every call.)
    This wrapper makes encode/decode trivial - just cal .encode('hello') and get tokens back.

    It also handles the EOS token consistently so we never accidentally forget to add it during training data prep.

    """

    def __init__(self, config: TokenizerConfig=None):
        """
        Initialize the tokenizer with GPT-2's BPE vocabulary.
        We use a pretrained tokenizer because:
            1. Training a tokenizer from scratch takes weeks of CPU time.
            2. GPT-2's tokenizer is open source, fast and well tested.
            3. Using the same tokenizer as production models means our code works identically to how GPT-3 tokenizes
        """

        self.config = config or TokenizerConfig() # initializes to TokenizerConfig() if config is None.
        
        """
        Load the GPT-2 encoding from tiktoken
        tiktoken stores the pretrained BPE merge tables.
        get_encoding('gpt2') loads the exact 50k merges that GPT-2 was trained with.
        """
        self.enc = tiktoken.get_encoding(self.config.name)

        """
        Define and encode the End-Of-Sequence token
        <|endoftext|> is the special token that marks boundaries between documents.
        During the training, we insert it between every document so the model learns where one text ends and another begins.
        """

        self.eos_token = '<|endoftext|>'       # the string representation of eos
        self.eos_token_id = self.enc.encode(   # convert to its token id
            self.eos_token,                    
            allowed_special={self.eos_token},   # tiktoken blocks special tokens by default safety. we must explicitly allow EOS encoding
        )[0]                                    # [0] because encode() returns a list - we want the single ID.


    def encode(self, text: str) -> List[int]:
        """
        Turn text into a list of integer token ID's
        Neural networks only eat numbers. Raw strings like 'Hello Worls' mean nothing to matrix multiplications.

        Example: 'Hello World' -> [15496, 995]

        Under the hood: tiktoken splits the text into subword pieces using the pretrained BPE merge table, then look up
        each piece's ID in the vocabulary.
        """
        encoded_data = self.enc.encode(text, allowed_special={self.eos_token})   # tiktoken's is written in Rust, not Python. It can tokenize hundreds of MB of text per second. A Pure python BPE tokenizer would be 100x slower.
        return encoded_data
    
    def decode(self, ids: List[int]) -> str:
        """
        Turn token ID's back into human readable text
        After the model generates a sequence of token IDs during the inference,
        we need to convert them back to text so humans can read the output.

        Example: [15496, 995] -> 'Hello World'
        """
        return self.enc.decode(ids)
    

    @property
    def vocab_size(self) -> int:
        """
        how many unique tokens exist in the vocabulary.
        this number determined the size of our model's output layer - the final linear layer must have
        vocab_size outputs (one score for each possible next token).

            50,257 means the model chooses from 50,257 possibilities every time it predicst the next word.
        """ 
        return self.config.vocab_size
    


if __name__ == '__main__':
    tokenizer = SimpleTokenizer()

    # test 1: basic test
    test_text = 'The cat sat on the mat'
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    print(f"Test 1 — Basic:")
    print(f"  Original: '{test_text}'")
    print(f"  Encoded:  {encoded}")
    print(f"  Decoded:  '{decoded}'")
    print(f"  Match:    {test_text == decoded}")

    # Test 2: EOS token
    eos = tokenizer.encode(tokenizer.eos_token)
    print(f"\nTest 2 — EOS token:")
    print(f"  String: '{tokenizer.eos_token}'")
    print(f"  Token ID: {tokenizer.eos_token_id}")
    print(f"  Encode result: {eos}")

    # Test 3: Rare/unseen word
    rare = tokenizer.encode("antidisestablishmentarianism")
    decoded_rare = tokenizer.decode(rare)
    print(f"\nTest 3 — Rare word:")
    print(f"  Encoded: {rare}")
    print(f"  Pieces:  {[tokenizer.decode([t]) for t in rare]}")
    print(f"  Decoded: '{decoded_rare}'")

    # Test 4: Emoji/Unicode
    emoji = tokenizer.encode("Hello 😊 world")
    print(f"\nTest 4 — Emoji:")
    print(f"  Encoded: {emoji}")
    print(f"  Decoded: '{tokenizer.decode(emoji)}'")

    print(f"\n  Vocab size: {tokenizer.vocab_size:,}")