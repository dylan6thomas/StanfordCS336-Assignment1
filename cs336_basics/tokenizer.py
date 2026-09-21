# Tokenizer class for encoding and decoding raw text
from collections.abc import Iterable, Iterator
import regex as re
import json
from collections import defaultdict
import heapq
from tests.common import gpt2_unicode_to_bytes

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
  """
  Tokenizer class that, given a vocabulary
  and a list of merges, encodes text into
  integer IDs and decodes integer IDs into
  text. Also supports user provided special
  tokens.
  """
  def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
    self.id_to_token = vocab
    self.token_to_id = {v: k for k,v in vocab.items()}
    self.merges = merges
    if special_tokens:
      self.special_tokens = set([special_token.encode() for special_token in special_tokens])
    else:
      self.special_tokens = []

  @classmethod
  def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
    """
    Constructs and returns Tokenizer from a
    serialized vocabulary and list of merges
    (in the same format as BPY training output)
    and a list of special tokens.
    """

    vocab = {}
    merges = []
    unicode_to_bytes = gpt2_unicode_to_bytes()

    with open(vocab_filepath, "r") as f:
      str_vocab = json.load(f)
    with open(merges_filepath, "r") as f:
      str_merges = json.load(f)

    for id, str_token in str_vocab.items():
      vocab[id] = bytes([unicode_to_bytes[c] for c in str_token])
    for str_merge in str_merges:
      s1, s2 = str_merge
      b1 = bytes([unicode_to_bytes[c] for c in s1])
      b2 = bytes([unicode_to_bytes[c] for c in s2])
      merges.append((b1, b2))

    if special_tokens:
      for special_token in special_tokens:
        if special_token.encode() not in vocab.values():
          vocab[len(vocab)] = special_token.encode()

    return cls(vocab, merges, special_tokens)
    
  def encode(self, text: str) -> list[int]:
    """Encode an input text into a sequence of token IDs"""

    pretoken_to_sequence = {}
    pair_to_pretokens = defaultdict(set)

    if self.special_tokens:
      splits = [re.escape(special_token.decode()) for special_token in self.special_tokens]

      splits.sort(key=len, reverse=True)

      pattern = f"({"|".join(splits)})"

      raw_sections = re.split(pattern, text)

      sections = [t for t in raw_sections if t != ""]

    else:
      sections = [text]

    pretoken_sequence = []

    for section in sections:
      if section.encode() in self.special_tokens:
        pretoken_sequence.append(section.encode())
      else:
        pretokens = re.finditer(PAT, section)
        for match in pretokens:
          pretoken = match.group().encode()
          pretoken_sequence.append(pretoken)
          if pretoken not in pretoken_to_sequence:
            sequence = [bytes([b]) for b in pretoken]
            pretoken_to_sequence[pretoken] = sequence
            for i in range(len(sequence)-1):
              pair_to_pretokens[(sequence[i], sequence[i+1])].add(pretoken)
        

    out = []

    merge_to_priority = {self.merges[i]: i for i in range(len(self.merges))}
    merge_candidates = set(self.merges)
    next_merge = [(merge_to_priority[merge], merge) for merge in self.merges]

    while next_merge:
      _, merge= heapq.heappop(next_merge)
      new_token = b''.join(merge)
      merge_candidates.remove(merge)
      for pretoken in pair_to_pretokens[merge]:
        s = pretoken_to_sequence[pretoken]
        i = 0
        while i < len(s)-1:
          if (s[i], s[i+1]) == merge:
            s[i] = new_token
            s.pop(i+1)
            if i < len(s) - 1:
              pair_to_pretokens[(new_token, s[i+1])].add(pretoken)
              front = (new_token, s[i+1])
              if front not in merge_candidates and front in merge_to_priority:
                merge_candidates.add(front)
                heapq.heappush(next_merge, (merge_to_priority[front], front))
            if i > 0:
              pair_to_pretokens[(s[i-1], new_token)].add(pretoken)
              back = (s[i-1], new_token)
              if back not in merge_candidates and back in merge_to_priority:
                merge_candidates.add(back)
                heapq.heappush(next_merge, (merge_to_priority[back], back))
          i += 1

    for pretoken in pretoken_sequence:
      if pretoken in self.special_tokens:
        out.append(self.token_to_id[pretoken])
      else:
        for token in pretoken_to_sequence[pretoken]:
          out.append(self.token_to_id[token])

    return out



    
  def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
    """
    Given an iterable of strings, return a generator
    that lazily yields token IDs
    """
    for s in iterable:
      yield from self.encode(s)

  def decode(self, ids: list[int]) -> str:
    """Decode a sequence of token IDs into text"""
    encoded_str = b"".join([self.id_to_token[idx] for idx in ids])
    return encoded_str.decode(errors="surrogateescape")