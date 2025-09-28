from transformers import GPT2TokenizerFast


class GPT2TokenizerWrapper:
    EOS = None

    def __init__(self, add_eos: bool = True):
        self.add_eos = add_eos

        tok = GPT2TokenizerFast.from_pretrained("gpt2")
        # Ensure an EOS token exists; GPT-2 uses <|endoftext|> (id 50256)
        if tok.eos_token_id is None:
            tok.add_special_tokens({"eos_token": "<|endoftext|>"})

        # So lift the advisory cap
        tok.model_max_length = int(1e9)

        self._tok = tok
        self.EOS = tok.eos_token_id
        self.eos_token_id = self.EOS
        self.vocab_size = tok.vocab_size

        self.special_token_map = {self.EOS: "<|endoftext|>"}

    def __len__(self) -> int:
        return self.vocab_size

    def encode(self, text: str) -> list[int]:
        ids = self._tok.encode(text, add_special_tokens=False)
        if self.add_eos:
            ids.append(self.EOS)
        return ids

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()

        if skip_special_tokens:
            # Strip EOS and decode the rest
            filtered = [i for i in ids if i != self.EOS]
            return self._tok.decode(filtered, clean_up_tokenization_spaces=True)

        parts = []
        span = []

        def flush():
            if span:
                parts.append(self._tok.decode(span, clean_up_tokenization_spaces=True))
                span.clear()

        for i in ids:
            if i == self.EOS:
                flush()
                parts.append(self.special_token_map.get(self.EOS, "<eos>"))
            else:
                span.append(i)
        flush()
        return "".join(parts)
