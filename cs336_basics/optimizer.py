import torch
from torch import nn
from typing import Optional, Callable
import math

class AdamW(torch.optim.Optimizer):
  def __init__(self, params, lr, betas, eps, weight_decay):
    defaults = {'lr': lr, 'beta1': betas[0], 'beta2': betas[1],
                'eps': eps, 'weight_decay': weight_decay}

    super().__init__(params, defaults)

  def step(self, closure: Optional[Callable] = None):
    loss = None if closure is None else closure()

    for group in self.param_groups:
      lr = group['lr']
      beta1 = group['beta1']
      beta2 = group['beta2']
      eps = group['eps']
      weight_decay = group['weight_decay']

      for p in group['params']:
        if p.grad is None:
          continue
        state = self.state[p]
        t = state.get('t', 1)
        m = state.get('m', 0)
        v = state.get('v', 0)

        adjusted_lr = lr * torch.div(math.sqrt(1-beta2**t), 1-beta1**t)

        grad = p.grad.data

        p.data -= lr * weight_decay * p.data
        m = beta1*m + (1-beta1)*grad
        v = beta2*v + (1-beta2)*grad**2

        p.data -= adjusted_lr*torch.div(m, torch.sqrt(v)+eps)

        state['t'] = t + 1
        state['m'] = m
        state['v'] = v

    return loss


def lr_cosine_schedule(t, a_max, a_min, t_warmup, t_cosine):
  if t < t_warmup:
    return t * a_max / t_warmup
  if t_warmup <= t and t <= t_cosine:
    return a_min + 0.5 * (1 + math.cos((t - t_warmup)/(t_cosine - t_warmup) * math.pi)) * (a_max - a_min)
  else:
    return a_min

def gradient_clipping(params, max_norm, eps=1e-6):
  norm = torch.nn.utils.get_total_norm([p.grad for p in params if p.grad is not None])

  if norm >= max_norm:
    for p in params:
      if p.grad is not None:
        p.grad.data *= (max_norm / (norm + eps))

  return params

