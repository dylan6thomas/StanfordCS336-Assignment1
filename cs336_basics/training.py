import torch
from torch import nn
import numpy as np
import os
import typing
from cs336_basics.transformer import TransformerLM
from cs336_basics.optimizer import AdamW, gradient_clipping, lr_cosine_schedule
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.loss import cross_entropy_loss
import json

def get_batch(x: np.typing.NDArray, batch_size: int, context_length: int, device='mps'):

  indices = np.random.randint(0, len(x) - context_length, (batch_size, ))

  in_sequence = torch.stack([torch.tensor(x[i:i+context_length]) for i in indices]).to(device)
  target = torch.stack([torch.tensor(x[i+1:i+context_length+1]) for i in indices]).to(device)

  return in_sequence, target

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
  model_dict = model.state_dict()
  optimizer_dict = optimizer.state_dict()

  final_dict = {'model': model_dict, 'optimizer': optimizer_dict, 'iteration': iteration}

  torch.save(final_dict, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
  state = torch.load(src)
  model_dict = state['model']
  optim_dict = state['optimizer']

  model.load_state_dict(model_dict)
  optimizer.load_state_dict(optim_dict)

  return state['iteration']

def train_model(
    config
):

    model_config = config['model']
    optimizer_config = config['optimizer']
    encoding_config = config['encoding']
    
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

    data = np.memmap(encoding_config["encoded_data"], np.int32)

    for i in range(config["steps"]):
       optimizer.zero_grad()

       input, target = get_batch(data, model_config["batch_size"], model_config["context_length"], model_config["device"])
       pred = model.forward(input)

       loss = cross_entropy_loss(pred, target)

       print(loss.cpu().item())

       loss.backward()

       for group in optimizer.param_groups:
        group["params"] = gradient_clipping(group["params"], optimizer_config["max_norm"])
        group["lr"] = lr_cosine_schedule(
          i,
          optimizer_config["a_max"],
          optimizer_config["a_min"],
          optimizer_config["t_warmup"],
          optimizer_config["t_cosine"]
        )

       optimizer.step()
       save_checkpoint(model, optimizer, i, config["save_path"])

if __name__ == "__main__":
  config_path = "/Users/dylanthomas/Documents/StanfordCS336/StanfordCS336-Assignment1/config.json"
  with open(config_path, "r") as c:
    config = json.load(c)

  train_model(config) 