from cs336_basics.inference import decode
import torch
from torch import nn
import numpy as np
import os
import typing
from cs336_basics.transformer import TransformerLM
from cs336_basics.optimizer import AdamW, gradient_clipping, lr_cosine_schedule
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.loss import cross_entropy_loss
from cs336_basics.training import load_checkpoint
import json

import sys

def train_model(
    config
):
    
    model_config = config['model']
    optimizer_config = config['optimizer']
    encoding_config = config['encoding']

    model_path = config["load_path"]
    prompt = config["prompt"]
    
    model = TransformerLM(
        model_config["vocab_size"],
        model_config["context_length"],
        model_config["d_model"],
        model_config["num_layers"],
        model_config["num_heads"],
        model_config["d_ff"],
        model_config["rope_theta"],
        model_config["device"]
    )

    optimizer = AdamW(
       model.parameters(),
       optimizer_config["lr"],
       optimizer_config["betas"],
       optimizer_config["eps"],
       optimizer_config["weight_decay"]
    )

    tokenizer = Tokenizer.from_files(
       encoding_config["vocab_path"],
       encoding_config["merges_path"],
       ['<|endoftext|>']
    )

    load_checkpoint(model_path, model, optimizer)

    print(decode(model, tokenizer, prompt, 1000))


if __name__ == "__main__":
  config_path = "/Users/dylanthomas/Documents/StanfordCS336/StanfordCS336-Assignment1/config.json"
  with open(config_path, "r") as c:
    config = json.load(c)

  sys.stdout.reconfigure(errors="ignore")

  train_model(config) 