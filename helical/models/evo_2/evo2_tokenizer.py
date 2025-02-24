# based on https://github.com/EleutherAI/gpt-neox/blob/main/megatron/tokenizer/tokenizer.py
from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
import torch


class AbstractTokenizer(ABC):
    """Abstract class for tokenizer."""

    def __init__(self, name):
        self.name = name
        super().__init__()

    @property
    @abstractmethod
    def vocab_size(self):
        pass

    @property
    @abstractmethod
    def vocab(self):
        """Dictionary from vocab text token to id token."""
        pass

    @property
    @abstractmethod
    def inv_vocab(self):
        """Dictionary from vocab id token to text token."""
        pass

    @abstractmethod
    def tokenize(self, text):
        pass

    def detokenize(self, token_ids):
        raise NotImplementedError(
            "detokenizer is not implemented for {} " "tokenizer".format(self.name)
        )

    @property
    def cls(self):
        raise NotImplementedError(
            "CLS is not provided for {} " "tokenizer".format(self.name)
        )

    @property
    def sep(self):
        raise NotImplementedError(
            "SEP is not provided for {} " "tokenizer".format(self.name)
        )

    @property
    def pad(self):
        raise NotImplementedError(
            "PAD is not provided for {} " "tokenizer".format(self.name)
        )

    @property
    def eod(self):
        raise NotImplementedError(
            "EOD is not provided for {} " "tokenizer".format(self.name)
        )

    @property
    def mask(self):
        raise NotImplementedError(
            "MASK is not provided for {} " "tokenizer".format(self.name)
        )


class CharLevelTokenizer(AbstractTokenizer):
    """Character Level Tokenizer"""

    def __init__(self, vocab_size):
        name = "CharLevelTokenizer"
        super().__init__(name)
        self._vocab_size = vocab_size
        self.eod_id = 0
        self.eos_id = 0
        self.pad_id = 1

    def clamp(self, n):
        return max(32, min(n, self.vocab_size))

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def vocab(self):
        raise NotImplementedError

    @property
    def inv_vocab(self):
        raise NotImplementedError

    def decode_token(self, token: int):
        return str(chr(self.clamp(token)))

    def tokenize(self, text: str):
        return list(np.fromstring(text, dtype=np.uint8))

    def tokenize_batch(self, text_batch: Union[List[str], str]):
        if isinstance(text_batch, list):
            return [self.tokenize(s) for s in text_batch]
        else:
            return self.tokenize(text_batch)

    def detokenize(self, token_ids):
        return "".join(list(map(self.decode_token, token_ids)))

    def detokenize_batch(self, token_ids: Union[List[str], str]):
        if isinstance(token_ids, list):
            return [self.detokenize(s) for s in token_ids]
        # elif if tensor, convert to list first
        elif isinstance(token_ids, torch.Tensor):
            return [self.detokenize(s) for s in token_ids.tolist()]
        else:
            return self.detokenize(token_ids)

    @property
    def eod(self):
        return self.eod_id

    # duplicate to suppose both names, eos and eod
    @property
    def eos(self):
        return self.eod_id
