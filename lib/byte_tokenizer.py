class ByteTokenizer:
    """
    - Vocab: [PAD=0, BOS=1, EOS=2] + 256 byte IDs (3..258)
    - encode: UTF-8 -> byte IDs
    - decode:
        * skip_special_tokens=True  -> ignore PAD/BOS/EOS and decode only bytes
        * skip_special_tokens=False -> decode bytes and insert special-token strings
    """
    PAD, BOS, EOS = 0, 1, 2
    OFFSET = 3

    def __init__(self, add_bos: bool = False, add_eos: bool = True):
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.vocab_size = self.OFFSET + 256
        self.pad_token_id = self.PAD
        self.bos_token_id = self.BOS
        self.eos_token_id = self.EOS
        self.special_token_map = {
            self.PAD: "<pad>",
            self.BOS: "<bos>",
            self.EOS: "<eos>",
        }

    def __len__(self) -> int:
        return self.vocab_size

    def encode(self, text: str) -> list[int]:
        # Encode text to UTF-8 bytes and map to IDs [OFFSET..OFFSET+255]
        b = text.encode("utf-8", errors="strict")
        ids = [self.OFFSET + x for x in b]
        if self.add_bos:
            ids = [self.BOS] + ids
        if self.add_eos:
            ids = ids + [self.EOS]
        return ids

    def _is_byte_id(self, i: int) -> bool:
        # Check whether ID encodes a raw byte
        return self.OFFSET <= i < self.OFFSET + 256

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        # Accept lists, tuples, or Torch tensors
        if hasattr(ids, "tolist"):
            ids = ids.tolist()

        if skip_special_tokens:
            # Fast path: drop all specials, decode only raw bytes
            bytes_list = [i - self.OFFSET for i in ids if self._is_byte_id(i)]
            return bytes(bytes_list).decode("utf-8", errors="ignore")

        # Full path: insert textual markers for special tokens
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
                # Unknown IDs are silently ignored

        flush()
        return "".join(parts)
