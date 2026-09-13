import torch
from torch import nn
import math

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
  v = torch.max(x)

  zero_x = x - v

  exp_x = torch.exp(zero_x)

  exp_sum = torch.sum(exp_x, dim=dim, keepdim=True)

  return torch.div(exp_x, exp_sum)

def scaled_dot_product_atttention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None=None
):
  qk = torch.div(torch.einsum("... q d, ... k d -> ... q k", Q, K), math.sqrt(Q.shape[-1]))

  if mask is not None:
      qk[mask == 0] -= torch.inf

  qk = softmax(qk, -1)

  attention = torch.einsum('... q k, ... k d -> ... q d', qk, V)

  return attention

class Linear(nn.Module):
  def __init__(self, in_features, out_features, device=None, dtype=None):
    super().__init__()

    std = torch.sqrt(torch.div(2, in_features+out_features))
    weight = torch.empty((out_features, in_features), device=device, dtype=dtype)

    nn.init.trunc_normal_(weight, 0, std, -3*std, 3*std)

    self.weight = nn.Parameter(weight)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return torch.einsum("... j,k j-> ... k", x, self.weight)

class Embedding(nn.Module):
  def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
    super().__init__()
    embedding = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)

    nn.init.trunc_normal_(embedding, 0, 1, -3, 3)

    self.weight = nn.Parameter(embedding)

    self.num_embeddings = num_embeddings
    self.dtype = dtype or torch.float32
    
  def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
    tokens_one_hot = nn.functional.one_hot(token_ids, self.num_embeddings).to(self.dtype)

    return torch.einsum('... j,jk -> ... k', tokens_one_hot, self.weight)

class RMSNorm(nn.Module):
  def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
    super().__init__()

    self.eps = eps
    self.d_model = d_model

    gain = torch.ones((d_model), device=device, dtype=dtype)
    self.gain = nn.Parameter(gain)



  def forward(self, x: torch.Tensor) -> torch.Tensor:

    in_type = x.dtype
    x_upcast = x.to(torch.float32)
    rms = torch.sqrt(torch.div(torch.sum((x_upcast**2), -1, keepdim=True), self.d_model) + self.eps)

    out = torch.div(x_upcast * self.gain, rms)
    out = out.to(in_type)

    return out

class SwiGLU(nn.Module):
  def __init__(self, d_model: int, d_ff=None, device=None, dtype=None):
    super().__init__()

    self.d_ff = d_ff or round((d_model*8/3)/64) * 64
    self.d_model = d_model

    self.w1 = Linear(self.d_model, self.d_ff)
    self.w2 = Linear(self.d_ff, self.d_model)
    self.w3 = Linear(self.d_model, self.d_ff)

  def forward(self, x: torch.Tensor) -> torch.Tensor:

    a = self.w1.forward(x)
    silu = torch.div(a, 1 + torch.exp(-a))

    b = silu * (self.w3.forward(x))

    out = self.w2.forward(b)

    return out

class RotaryPositionalEmbedding(nn.Module):
  def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
    super().__init__()

    positions = torch.arange(0, max_seq_len, device=device)
    frequencies = torch.arange(1, d_k/2+1, device=device)

    self.d_k = d_k

    inv_freq = torch.pow(theta,torch.div(-2*(frequencies-1),d_k))

    thetas = torch.outer(positions, inv_freq)

    self.register_buffer('thetas', thetas, False)


  def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:

    token_thetas = self.thetas[token_positions]

    token_sin = torch.sin(token_thetas)
    token_cos = torch.cos(token_thetas)

    x_pairs = x.reshape((x.shape[:-1]+(self.d_k//2, 2)))

    x_1 = x_pairs[..., 0]
    x_2 = x_pairs[..., 1]

    rotation_1 = x_1 * token_cos - x_2 * token_sin
    rotation_2 = x_1 * token_sin + x_2 * token_cos

    out = torch.stack((rotation_1, rotation_2), -1).reshape(x.shape)

    return out

class CausalMHSelfAttention(nn.Module):
  def __init__(self, d_model: int, num_heads: int, rope_theta: float, max_seq_len: int, device=None):
    super().__init__()

    self.d_model = d_model
    self.d_k = d_model // num_heads
    self.d_v = d_model // num_heads

    self.wq = Linear(num_heads * self.d_k, d_model)
    self.wk = Linear(num_heads * self.d_k, d_model)
    self.wv = Linear(num_heads * self.d_v, d_model)
    self.wo = Linear(d_model, num_heads * self.d_v)

    self.rope = RotaryPositionalEmbedding(rope_theta, self.d_k, max_seq_len, device)

    self.max_seq_len = max_seq_len

  def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:

    seq_len = x.shape[-2]
    mask = torch.tril(torch.ones(x.shape[:-2]+(self.max_seq_len, self.max_seq_len)))

    q = self.wq.forward(x)
    k = self.wk.forward(x)
    q_rope = torch.stack(
      [self.rope.forward(q[..., self.d_k*i:self.d_k*(i+1)], token_positions) 
        for i in range(self.d_model // self.d_k)], -1)
    k_rope = torch.stack(
      [self.rope.forward(k[..., self.d_k*i:self.d_k*(i+1)], token_positions) 
        for i in range(self.d_model // self.d_k)], -1)
    v = self.wv.forward(x)

    print(self.d_k)
    for i in range(self.d_model // self.d_k):
      print(q_rope[..., i].shape)
      print(k_rope[..., i].shape)
      print(v[..., self.d_k*i:self.d_k*(i+1)].shape)
    

    multihead = torch.concat([scaled_dot_product_atttention(
      q_rope[..., i],
      k_rope[..., i],
      v[..., self.d_v*i:self.d_v*(i+1)], 
      mask[:, :seq_len, :seq_len]) for i in range(self.d_model // self.d_k)], dim=-1)

    print(f"MULTIHEAD SHAPE: {multihead.shape}")

    return self.wo.forward(multihead)
