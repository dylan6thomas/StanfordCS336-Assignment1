import torch
from torch import nn
import math

from cs336_basics.transformer import softmax

def cross_entropy_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

  v = torch.max(logits, dim=-1, keepdim=True).values
  
  zero_logits = logits - v

  gathered = torch.gather(input=zero_logits, dim=-1, index=target.unsqueeze(-1))

  loss = -(gathered - torch.log(torch.sum(torch.exp(zero_logits), -1, keepdim=True)))

  return torch.mean(loss)