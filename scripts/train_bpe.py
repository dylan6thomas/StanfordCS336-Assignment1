from cs336_basics import bpe, tokenizer
import os

if __name__ == "__main__":

  data_root = "/Users/dylanthomas/Documents/StanfordCS336/StanfordCS336-Assignment1/data"

  vocab, merges = bpe.train_bpe(os.path.join(data_root, "TinyStoriesV2-GPT4-train.txt"), 10000, ["<|endoftext|>"])
  bpe.save_bpe(os.path.join(data_root, "vocab.json"),
              os.path.join(data_root, "merges.json"),
              vocab,
              merges)