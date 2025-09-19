class ByteTokenizer:
    """
    - vocab: [PAD=0, BOS=1, EOS=2] + 256 Byte-IDs (3..258)
    - encode: UTF-8 -> Byte-IDs
    - decode: Byte-IDs -> UTF-8-String (ignores PAD/BOS/EOS)
    """
    PAD, BOS, EOS = 0, 1, 2
    OFFSET = 3

    def __init__(self, add_bos: bool = False, add_eos: bool = True):
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.vocab_size = self.OFFSET + 256
        self.pad_token_id = self.PAD

    def __len__(self) -> int:
        return self.vocab_size

    def encode(self, text: str) -> list[int]:
        b = text.encode("utf-8", errors="strict")
        ids = [self.OFFSET + x for x in b]
        if self.add_bos:
            ids = [self.BOS] + ids
        if self.add_eos:
            ids = ids + [self.EOS]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        if skip_special_tokens:
            bytes_list = [
                i - self.OFFSET for i in ids
                if self.OFFSET <= i < self.OFFSET + 256
            ]
            return bytes(bytes_list).decode("utf-8", errors="ignore")

        special_map = {
            self.PAD: "<pad>",
            self.BOS: "<bos>",
            self.EOS: "<eos>",
        }

        parts: list[str] = []
        byte_buffer: list[int] = []

        def flush_buffer():
            if byte_buffer:
                parts.append(bytes(byte_buffer).decode("utf-8", errors="ignore"))
                byte_buffer.clear()

        for i in ids:
            if self.OFFSET <= i < self.OFFSET + 256:
                byte_buffer.append(i - self.OFFSET)
            else:
                flush_buffer()
                if i in special_map:
                    parts.append(special_map[i])

        flush_buffer()
        return "".join(parts)
