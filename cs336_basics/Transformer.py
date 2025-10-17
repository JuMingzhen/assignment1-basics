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
        flat_x = x.reshape(-1) 
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

class multihead_self_attention(nn.Module):
    def __init__(self, d_model: int, num_heads:int, weight: dict[str,Tensor] = None, 
                 RoPE = None, token_positions = None, device = None, dtype = None):
        super().__init__()
        self.h = num_heads
        self.d = d_model
        self.d_h = int(self.d / self.h) 
        self.device = device
        if weight is not None:
            self.W_k = nn.Parameter(weight["k"]).to(device=device, dtype = dtype)
            self.W_q = nn.Parameter(weight["q"]).to(device=device, dtype = dtype)
            self.W_v = nn.Parameter(weight["v"]).to(device=device, dtype = dtype)
            self.W_o = nn.Parameter(weight["o"]).to(device=device, dtype = dtype)
        else:
            self.W_k = nn.Parameter(torch.empty((d_model, d_model), device = device, dtype = dtype))
            self.W_q = nn.Parameter(torch.empty((d_model, d_model), device = device, dtype = dtype))
            self.W_v = nn.Parameter(torch.empty((d_model, d_model), device = device, dtype = dtype))
            self.W_o = nn.Parameter(torch.empty((d_model, d_model), device = device, dtype = dtype))
            sigma = torch.sqrt(torch.tensor(1.0 / d_model))
            nn.init.trunc_normal_(self.W_k, mean=0.0, std=sigma, a=-3*sigma, b=3*sigma)
            nn.init.trunc_normal_(self.W_q, mean=0.0, std=sigma, a=-3*sigma, b=3*sigma)
            nn.init.trunc_normal_(self.W_v, mean=0.0, std=sigma, a=-3*sigma, b=3*sigma)
            nn.init.trunc_normal_(self.W_o, mean=0.0, std=sigma, a=-3*sigma, b=3*sigma)
        self.RoPE = RoPE
        self.token_positions = token_positions
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        Q = einsum(self.W_q, x, "d_q dim, ... dim -> ... d_q")
        K = einsum(self.W_k, x, "d_k dim, ... dim -> ... d_k")
        V = einsum(self.W_v, x, "d_v dim, ... dim -> ... d_v")
        Q = rearrange(Q, "... seq (head d_h) -> ... head seq d_h", head = self.h)
        K = rearrange(K, "... seq (head d_h) -> ... head seq d_h", head = self.h)
        V = rearrange(V, "... seq (head d_h) -> ... head seq d_h", head = self.h)
        seq = Q.size(-2)
        if self.RoPE:
            if self.token_positions is not None:
                expanded_seq = self.token_positions
            else:
                prefix_dims = Q.shape[:-2]
                base_seq = torch.arange(seq, dtype=torch.int64, device = self.device)
                for _ in range(len(prefix_dims)):
                    base_seq = base_seq.unsqueeze(0)  # 在最前面插入新维度
                expanded_seq = base_seq.repeat(*prefix_dims, 1)
            K = self.RoPE(K, expanded_seq)
            Q = self.RoPE(Q, expanded_seq)
        mask = ~ torch.triu(torch.ones((seq, seq), device = self.device), diagonal=1).bool()
        Temp = scaled_dot_product_attention(K, Q, V, mask)
        Temp = rearrange(Temp, "... head seq d_h -> ... seq (head d_h)")
        return einsum(Temp, self.W_o, "... dim1, dim dim1 -> ... dim")

class transformer_block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, eps = None, RoPE = None,
                 token_positions = None, weights = None, device = None, dtype = None):
        '''
        weights: dict---> keys: "MHA", "RMS1", "RMS2", "ff"
        where each values of keys above are:
        weights["MHA"] = {'k': *, 'q': *,'v': *,'o': *}
        weights["RMS1"] = {'gamma': *}
        weights["RMS2"] = {'gamma': *}
        weights["ff"] = {'W1.W': *, 'W2.W': *, 'W3.W': *}
        '''
        super().__init__()
        self.d_model = d_model
        self.h = num_heads
        self.d_ff = d_ff
        weight_MHA = None
        if eps == None:
            self.RMS1 = RMSNorm(d_model, device = device, dtype = dtype)
            self.RMS2 = RMSNorm(d_model, device = device, dtype = dtype)
        else:
            self.RMS1 = RMSNorm(d_model, eps, device, dtype)
            self.RMS2 = RMSNorm(d_model, eps, device, dtype)
        self.ff = SwiGLU(d_model, d_ff, device, dtype)
        if weights is not None:
            weight_MHA = weights["MHA"]
            self.RMS1.load_state_dict(weights['RMS1'])
            self.RMS2.load_state_dict(weights['RMS2'])
            self.ff.load_state_dict(weights['ff'])
        self.MHA = multihead_self_attention(d_model, num_heads, weight_MHA, RoPE, 
                                            token_positions, device, dtype)
    def forward(self, x: torch.Tensor):
        x = x + self.MHA(self.RMS1(x))
        x = x + self.ff(self.RMS2(x))
        return x
    
class transformer_lm(nn.Module):
    """
    weights: dict---> keys: "MHA", "embedding", "final_ff", "final_norm"
    where each values of keys above are:
    weights["MHA"] = list[{'MHA.W_k': *, 'MHA.W_q': *,'MHA.W_v': *,'MHA.W_o': *,
      'RMS1.gamma': *, 'RMS2.gamma': *, 'ff.W1.W': *, 'ff.W2.W': *, 'ff.W3.W': *}]
    weights["embedding"] = {'embedding': *}
    weights["final_norm"] = {'gamma': *}
    weights["final_ff"] = {'W': *}
    """
    def __init__(self, vocab_size:int, context_length:int, num_layers:int, d_model:int,
                 num_heads = 4, d_ff = 128, eps:float = 0.00001, RoPE_theta = 10000,  
                 weights = None, device = None, dtype = None):
        super().__init__()
        self.num_heads = num_heads
        self.max_length = context_length
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_ff = d_ff

        self.embedding = Embedding(vocab_size, d_model, device, dtype)
        self.norm = RMSNorm(d_model, eps, device, dtype)
        self.final_ff = Linear(d_model, vocab_size, device, dtype)
        RoPE = RotaryPositionalEmbedding(RoPE_theta, int(d_model/num_heads), context_length, device, dtype)
        self.transformer_blocks = nn.ModuleList([
            transformer_block(d_model, num_heads, d_ff, eps, RoPE, None, None, device, dtype)
            for _ in range(num_layers)
        ])
        if weights is not None:
            self.embedding.load_state_dict(weights["embedding"])
            self.norm.load_state_dict(weights["final_norm"])
            self.final_ff.load_state_dict(weights["final_ff"])
            for _ in range(num_layers):
                self.transformer_blocks[_].load_state_dict(weights["MHA"][_])
        
    def forward(self, x:torch.Tensor):
        assert x.size(-2) <= self.max_length
        activation = self.embedding(x)
        for module in self.transformer_blocks:
            activation = module(activation)
        return self.final_ff(self.norm(activation))
    
