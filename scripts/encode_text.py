from cs336_basics.tokenizer import Tokenizer
import os
import itertools
import numpy as np

if __name__ == "__main__":
  root = "/Users/dylanthomas/Documents/StanfordCS336/StanfordCS336-Assignment1/data"
  vocab_path = os.path.join(root, "vocab.json")
  merges_path = os.path.join(root, "merges.json")
  out_path = os.path.join(root, "owt_val.bin")
  tokenizer = Tokenizer.from_files(vocab_path, merges_path, ["<|endoftext|>"])

  with open(os.path.join(root, "owt_valid.txt"), "r") as f:
    encodings = tokenizer.encode_iterable(f)
    batches = itertools.batched(encodings, 1000)

    with open(out_path, "w+b") as encoding_file:
      for batch in batches:
        batch_array = np.array(batch, dtype=np.int32)
        encoding_file.write(batch_array)

