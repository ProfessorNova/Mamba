import json
from typing import Dict, List


class SimpleTokenizer:
    """A simple character‑level tokenizer."""
    def __init__(self, text: str) -> None:
        # build vocabulary from unique characters in the text
        unique_chars = sorted(set(text))
        # reserve id 0 for unknown token
        self.unk_token = '<unk>'
        self.unk_id = 0
        # assign ids starting at 1 for known characters
        self.char_to_id: Dict[str, int] = {self.unk_token: self.unk_id}
        next_id = 1
        for ch in unique_chars:
            # skip newline by mapping it explicitly; we still include it in vocab
            if ch not in self.char_to_id:
                self.char_to_id[ch] = next_id
                next_id += 1
        # inverse mapping
        self.id_to_char: Dict[int, str] = {i: ch for ch, i in self.char_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.char_to_id)

    def encode(self, text: str) -> List[int]:
        # map each character to its id, using unk_id for unknown characters
        return [self.char_to_id.get(ch, self.unk_id) for ch in text]

    def decode(self, ids: List[int]) -> str:
        # map ids back to characters; unknown ids become '?' for readability
        chars = [self.id_to_char.get(i, '?') for i in ids]
        return ''.join(chars)

    def save(self, path: str) -> None:
        # save mappings to a JSON file
        data = {
            'unk_token': self.unk_token,
            'unk_id': self.unk_id,
            'char_to_id': self.char_to_id,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> 'SimpleTokenizer':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        obj = cls.__new__(cls)
        obj.unk_token = data['unk_token']
        obj.unk_id = data['unk_id']
        obj.char_to_id = {k: int(v) for k, v in data['char_to_id'].items()}
        obj.id_to_char = {i: ch for ch, i in obj.char_to_id.items()}
        return obj
