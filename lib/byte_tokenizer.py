class ByteTokenizer:
    EOS = 0
    OFFSET = 1

    def __init__(self, add_eos: bool = True):
        self.add_eos = add_eos
        self.vocab_size = self.OFFSET + 256  # 257
        self.eos_token_id = self.EOS
        self.special_token_map = {self.EOS: "<eos>"}

    def __len__(self) -> int:
        return self.vocab_size

    def _is_byte_id(self, i: int) -> bool:
        # Check whether ID encodes a raw byte
        return self.OFFSET <= i < self.OFFSET + 256

    def encode(self, text: str) -> list[int]:
        # Encode text to UTF-8 bytes and map to IDs [1..256]
        b = text.encode("utf-8", errors="strict")
        ids = [self.OFFSET + x for x in b]
        if self.add_eos:
            ids.append(self.EOS)
        return ids

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        # Accept lists, tuples, or Torch tensors
        if hasattr(ids, "tolist"):
            ids = ids.tolist()

        if skip_special_tokens:
            bytes_list = [i - self.OFFSET for i in ids if self._is_byte_id(i)]
            return bytes(bytes_list).decode("utf-8", errors="ignore")

        parts: list[str] = []
        byte_buf: list[int] = []

        def flush():
            if byte_buf:
                parts.append(bytes(byte_buf).decode("utf-8", errors="ignore"))
                byte_buf.clear()

        for i in ids:
            if self._is_byte_id(i):
                byte_buf.append(i - self.OFFSET)
            else:
                flush()
                tok = self.special_token_map.get(i)
                if tok is not None:
                    parts.append(tok)
        flush()
        return "".join(parts)
