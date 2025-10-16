import torch
from torch import Tensor 
import torch.nn as nn
from torch.optim import Optimizer 
from einops import rearrange, einsum, reduce, repeat
import math

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
        theta = 1.0 / (theta ** (torch.arange(0, d_k, 2, device = device, dtype=dtype) / d_k))
        seq_idx = torch.arange(max_seq_len, device = device, dtype=dtype)
        m_theta = torch.outer(seq_idx, theta)
        self.register_buffer("cos", torch.cos(m_theta), persistent=False)
        self.register_buffer("sin", torch.sin(m_theta), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # Algorithm reference: https://www.zhihu.com/tardis/bd/art/647109286?source_id=1001
        seq_len = x.size(-2)
        if seq_len is not None:
            cos = self.cos[:seq_len, :]
            sin = self.sin[:seq_len, :]
        cos = repeat(cos, "seq dim -> seq (dim repeat)", repeat = 2)
        sin = repeat(sin, "seq dim -> seq (dim repeat)", repeat = 2)
        sin = sin[token_positions]
        cos = cos[token_positions]
        # rearrange last dim of x
        # How to optimize codes below?
        x_s = rearrange(x, "... (d r) -> ... d r", r = 2)
        x_s = torch.flip(x_s, dims=[-1])
        x_s = rearrange(x_s, "... d r -> ... (d r)")
        minus = torch.tensor([-1.0, 1.0], device = self.device)
        minus = repeat(minus, "d -> (r d)", r = int(self.dim/2))
        x_s = einsum(x_s, minus, "... dim, dim -> ... dim")

        # calculate RoPE
        x = einsum(x, cos, "... dim, ... dim -> ... dim")
        x_s = einsum(x_s, sin, "... dim, ... dim -> ... dim")
        return x + x_s
    
def Softmax(x: torch.Tensor, dim: int):
    # Find the maximum value in the specified dimension for numerical stability
    max_vals = torch.max(x, dim=dim, keepdim=True).values
    # Subtract max values to prevent overflow
    stabilized = x - max_vals
    # Compute exponentials
    exponentials = torch.exp(stabilized)
    # Compute sum of exponentials along the specified dimension
    exp_sum = torch.sum(exponentials, dim=dim, keepdim=True)
    return exponentials / exp_sum

def scaled_dot_product_attention(k: torch.Tensor, q: torch.Tensor, 
                                 v: torch.Tensor, mask: torch.Tensor = None):
    d_k = math.sqrt(k.size(-1))
    pre_softmax = einsum(q, k, "... seq1 d_k, ... seq2 d_k -> ... seq1 seq2")
    pre_softmax = pre_softmax / d_k
    if mask is not None:
        pre_softmax = torch.where(mask, pre_softmax, float("-inf")) 
        # Be careful of using torch.masked_fill
    softmax = Softmax(pre_softmax, -1)
    return einsum(softmax, v, "... seq1 seq2, ... seq2 d_v -> ... seq1 d_v")

class multihead_self_attention(nn.Modules):
    def __init__(self, d_model: int, num_heads, device = None, dtype = None):
        super().__init__()
        self.h = num_heads
        self.W_k = nn.Parameter(torch.empty((d_model, d_model), device = device, dtype = dtype))
        self.W_q = nn.Parameter(torch.empty((d_model, d_model), device = device, dtype = dtype))
        self.W_v = nn.Parameter(torch.empty((d_model, d_model), device = device, dtype = dtype))
        self.W_o = nn.Parameter(torch.empty((d_model, d_model), device = device, dtype = dtype))
        
    def foward(self, x: torch.Tensor) -> torch.Tensor: 
        
        pass