# This is a Byte Pair Encoding Tokenizer
# This is trained by grouping the most common token groups, beginning with the original 256

import regex as re
import os
from typing import BinaryIO
from collections import Counter, defaultdict
import multiprocessing
from tests.common import gpt2_bytes_to_unicode
import json

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def get_chunk_pretokens(chunk, special_tokens):
  pretoken_to_freq = defaultdict(int)
  splits = [re.escape(special_token) for special_token in special_tokens]
  sections = re.split("|".join(splits), chunk)
  for section in sections:
    pretokens = re.finditer(PAT, section)
    for match in pretokens:
      pretoken = match.group().encode()
      pretoken_to_freq[pretoken] += 1

  return pretoken_to_freq

def process_chunk(input_path, start, end, special_tokens):
    """Worker opens the file itself and reads only its slice."""
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_text = f.read(end - start).decode("utf-8", errors="ignore")
    return get_chunk_pretokens(chunk_text, special_tokens)

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

  vocab = {i: bytes([i]) for i in range(256)}

  pretoken_to_sequence = {}
  pair_to_pretokens = defaultdict(lambda: defaultdict(int))
  pair_counts = defaultdict(int)
  pretoken_to_freq = defaultdict(int)
  merges = []

  with open(input_path, "rb") as f:
    num_processes = 2
    boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

  # Build args WITHOUT reading any text — just the byte offsets
  args = [
      (input_path, start, end, special_tokens)
      for start, end in zip(boundaries[:-1], boundaries[1:])
  ]

  pretoken_to_freq = Counter()
  with multiprocessing.Pool(num_processes) as pool:
      # imap streams results back one at a time instead of waiting for all of them
      for d in pool.starmap(process_chunk, args):
          pretoken_to_freq.update(d)

  for pretoken in pretoken_to_freq.keys():
      sequence = [bytes([c]) for c in pretoken]
      pretoken_to_sequence[pretoken] = sequence
      for i in range(len(sequence) - 1):
          pair_to_pretokens[(sequence[i], sequence[i + 1])][pretoken] += 1

  for pair, pretokens in pair_to_pretokens.items():
      for pretoken, count in pretokens.items():
          pair_counts[pair] += pretoken_to_freq[pretoken] * count

  while len(vocab) < vocab_size - len(special_tokens):
      highest_freq = max(pair_counts.values())
      options = [k for k, v in pair_counts.items() if v == highest_freq]

      merged_pair = max(options)

      new_token = b''.join(merged_pair)

      vocab[len(vocab)] = new_token
      merges.append(merged_pair)

      updated_pretokens = list(pair_to_pretokens[merged_pair].keys())

      for pretoken in updated_pretokens:
        seq = pretoken_to_sequence[pretoken]
        i = 0
        while i < len(seq) - 1:
          if (seq[i], seq[i+1]) == merged_pair:

            if i < len(seq) - 2:
              pair_counts[(seq[i+1], seq[i+2])] -= pretoken_to_freq[pretoken]
              pair_to_pretokens[(seq[i+1], seq[i+2])][pretoken] -= 1
            if i > 0:
              pair_counts[(seq[i-1], seq[i])] -= pretoken_to_freq[pretoken]
              pair_to_pretokens[(seq[i-1], seq[i])][pretoken] -= 1
            seq[i] = new_token

            seq.pop(i+1)
            pair_counts[merged_pair] -= pretoken_to_freq[pretoken]

            if i < len(seq) - 1:
              pair_counts[(new_token, seq[i+1])] += pretoken_to_freq[pretoken]
              pair_to_pretokens[(new_token, seq[i+1])][pretoken] += 1
            if i > 0:
              pair_counts[(seq[i-1], new_token)] += pretoken_to_freq[pretoken]
              pair_to_pretokens[(seq[i-1], new_token)][pretoken] += 1

          i += 1
        pair_to_pretokens[merged_pair][pretoken] = 0


  for special_token in special_tokens:
     vocab[len(vocab)] = special_token.encode()

  return vocab, merges

def save_bpe(vocab_save_path: str, merges_save_path: str, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]) -> None:
  byte_2_unicode = gpt2_bytes_to_unicode()
  int2str = {}
  str_merges = []

  for int, bytes in vocab.items():
    token = "".join([byte_2_unicode[byte] for byte in bytes])
    int2str[int] = token

  for merge in merges:
      t1, t2 = merge
      s1 = "".join([byte_2_unicode[byte] for byte in t1])
      s2 = "".join([byte_2_unicode[byte] for byte in t2])
      str_merges.append((s1, s2))

  with open(vocab_save_path, "w") as f:
    json.dump(int2str, f)
  with open(merges_save_path, "w") as f:
      json.dump(str_merges, f)
  

    
                
