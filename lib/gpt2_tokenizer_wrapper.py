# gpt2_tokenizer_wrapper_tiktoken.py
import os
from typing import List, Iterable, Union

import tiktoken
import torch


class GPT2TokenizerWrapper:
    EOS = 50256  # GPT-2 <|endoftext|>

    def __init__(self, add_eos: bool = True):
        self.add_eos = add_eos
        self._enc = tiktoken.get_encoding("gpt2")

        # Public attrs to match your original wrapper
        self.eos_token_id = self.EOS
        self.vocab_size = self._enc.n_vocab
        self.special_token_map = {self.EOS: "<|endoftext|>"}

    def __len__(self) -> int:
        return self.vocab_size

    # ---------- Encoding ----------
    def encode(self, text: str) -> List[int]:
        ids = self._enc.encode_ordinary(text or "")
        if self.add_eos:
            ids.append(self.EOS)
        return ids

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """
        Fast batch tokenization via tiktoken's Rust backend.
        Mirrors encode(): appends EOS if add_eos=True; otherwise returns raw ids.
        """
        ids_batch = self._enc.encode_ordinary_batch([t or "" for t in texts])
        if self.add_eos:
            return [ids + [self.EOS] for ids in ids_batch]
        return ids_batch

    # ---------- Decoding ----------
    def decode(self, ids: Union[Iterable[int], torch.Tensor], skip_special_tokens: bool = False) -> str:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()

        if skip_special_tokens:
            filtered = [i for i in ids if i != self.EOS]
            return self._enc.decode(filtered)

        parts: List[str] = []
        span: List[int] = []

        def flush_span():
            if span:
                parts.append(self._enc.decode(span))
                span.clear()

        for i in ids:
            if i == self.EOS:
                flush_span()
                parts.append(self.special_token_map.get(self.EOS, "<eos>"))
            else:
                span.append(i)
        flush_span()
        return "".join(parts)
