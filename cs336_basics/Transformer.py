import torch
from torch import Tensor 
import torch.nn as nn
from torch.optim import Optimizer 
from einops import rearrange, einsum, reduce, repeat

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.W = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        sigma = torch.sqrt(torch.tensor(2.0 / (in_features + out_features)))
        nn.init.trunc_normal_(self.W, mean=0.0, std=sigma, a=-3*sigma, b=3*sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Y = einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")
        return Y

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_num = num_embeddings
        self.embedding = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device = device, dtype = dtype)
        )
        nn.init.trunc_normal_(self.embedding, mean = 0.0, std = 1.0, a=-3.0, b=3.0)
    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        original_shape = x.shape
        flat_x = x.view(-1) 
        # (total_elements, embedding_dim)
        flat_embeds = torch.index_select(self.embedding, 0, flat_x)
        output_shape = original_shape + (self.embedding_dim,)
        output = flat_embeds.view(output_shape)
        return output

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.gamma = nn.Parameter(torch.ones(d_model, device = device, dtype = dtype))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        norm = x ** 2
        norm = reduce(norm, "batch seq dim -> batch seq", "mean")
        norm = torch.sqrt(norm + self.eps)
        norm = repeat(norm, "batch seq -> batch seq dim", dim = self.d_model)
        x = x / norm
        x = einsum(
                x, self.gamma,
                "batch seq dim, dim -> batch seq dim"
            )
        return x.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff = None, device = None, dtype = None):
        super().__init__()
        self.d_model = d_model
        if d_ff:
            self.d_ff = d_ff
        else:
            d_ff = 8 / 3 * d_model / 64
            d_ff = int(round(d_ff) * 64)
            self.d_ff = d_ff
        self.W1 = Linear(self.d_model, self.d_ff, device = device, dtype = dtype)
        self.W2 = Linear(self.d_ff, self.d_model, device = device, dtype = dtype)
        self.W3 = Linear(self.d_model, self.d_ff, device = device, dtype = dtype)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        SiLU = lambda t: t/(1 + torch.exp(-t))
        return self.W2(SiLU(self.W1(x))*self.W3(x))

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype = None):
        super().__init__()
        self.theta = theta
        self.dim = d_k
        self.max_len = max_seq_len
        self.device = device
        theta = 1.0 / (theta ** (torch.arange(0, d_k, 2) / d_k))
        seq_idx = torch.arange(max_seq_len)
        m_theta = torch.outer(seq_idx, theta)
        self.register_buffer("cos", torch.cos(m_theta, device = device, dtype=dtype), persistent=False)
        self.register_buffer("sin", torch.sin(m_theta, device = device, dtype=dtype), persistent=False)
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(-2)
        if seq_len is not None:
            cos = self.cos[:seq_len, :]
            sin = self.sin[:seq_len, :]
        cos = repeat(cos, "seq dim -> seq (dim repeat)", repeat = 2)
        sin = repeat(sin, "seq dim -> seq (dim repeat)", repeat = 2)
        # rearrange last dim
        dim = x.size(-1)
        x_reshaped = x.view(*x.shape[:-1], dim//2, 2)
        x_rearranged = torch.stack([
            -x_reshaped[..., 1], 
            x_reshaped[..., 0]    
        ], dim=-1)
        x_rearranged = x_rearranged.view(*x.shape)

