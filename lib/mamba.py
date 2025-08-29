import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MambaConfig:
    d_model: int
    n_layer: int
    vocab_size: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Optional[int] = 'auto'
    d_conv: int = 4
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False

    def __post_init__(self) -> None:
        # compute inner dimension
        self.d_inner: int = int(self.d_model * self.expand)
        # choose dt_rank automatically if needed
        if self.dt_rank == 'auto' or self.dt_rank is None:
            self.dt_rank = int(math.ceil(self.d_model / 16))
        # pad vocabulary size for performance
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            remainder = self.pad_vocab_size_multiple - (self.vocab_size % self.pad_vocab_size_multiple)
            self.vocab_size += remainder


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        # one weight per feature
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig) -> None:
        super().__init__()
        self.config = config
        d_model = config.d_model
        d_inner = config.d_inner
        n_state = config.d_state

        # project input to 2 * d_inner; will be split into x and residual/gate
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=config.bias)

        # depthwise 1‑D convolution on the x part
        # in_channels = out_channels = d_inner, groups=d_inner implements
        # depthwise convolution; padding ensures output length equals input.
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=config.d_conv,
            groups=d_inner,
            padding=config.d_conv - 1,
            bias=config.conv_bias,
        )

        # x_proj outputs (Δ, B, C); Δ has dimension dt_rank, B and C have n_state.
        self.x_proj = nn.Linear(d_inner, config.dt_rank + 2 * n_state, bias=False)
        # dt_proj projects Δ from dt_rank to d_inner
        self.dt_proj = nn.Linear(config.dt_rank, d_inner, bias=True)

        # initialise A as logarithm of negative values; shape (d_inner, n_state)
        # Using range 1..n_state for each inner channel as in the original paper.
        arange = torch.arange(1, n_state + 1).float()
        A_init = arange.log().unsqueeze(0).repeat(d_inner, 1)  # (d_inner, n_state)
        self.A_log = nn.Parameter(A_init)
        # D is a learnable diagonal term for the skip connection
        self.D = nn.Parameter(torch.ones(d_inner))

        # final projection back to d_model
        self.out_proj = nn.Linear(d_inner, d_model, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        b, l, _ = x.size()

        # input projection and split into x (for SSM) and residual gate
        projected = self.in_proj(x)  # (b, l, 2 * d_inner)
        x_part, res_part = projected.split(projected.size(-1) // 2, dim=-1)

        # depthwise convolution expects (b, channels, seq)
        x_conv = x_part.transpose(1, 2)  # (b, d_inner, l)
        x_conv = self.conv1d(x_conv)[:, :, :l]  # trim padding to original length
        x_conv = x_conv.transpose(1, 2)  # back to (b, l, d_inner)

        # activation
        x_conv = F.silu(x_conv)

        # selective state space model returns (b, l, d_inner)
        y = self.ssm(x_conv)

        # gating via residual part; apply SiLU before multiplication
        y = y * F.silu(res_part)

        # output projection
        out = self.out_proj(y)
        return out

    def ssm(self, u: torch.Tensor) -> torch.Tensor:
        b, l, d_in = u.size()
        n = self.config.d_state

        # compute fixed A and D; A = -exp(A_log)
        A = -torch.exp(self.A_log)  # (d_inner, n)
        D = self.D  # (d_inner)

        # project u to get Δ, B and C
        proj = self.x_proj(u)  # (b, l, dt_rank + 2*n)
        dt_rank = self.config.dt_rank
        delta_raw, B_raw, C_raw = proj.split([dt_rank, n, n], dim=-1)
        # Δ (b, l, d_inner); first project from dt_rank to d_inner then softplus
        delta = F.softplus(self.dt_proj(delta_raw))  # (b, l, d_inner)
        # B and C have shape (b, l, n)
        B = B_raw
        C = C_raw

        # discretise A and B using Δ; see Eq.(4) of the paper (zero‑order hold)
        # deltaA[t] = exp(Δ_t * A)
        # Here Δ is broadcasted over the n dimension of A.
        # Compute deltaA: (b, l, d_in, n) = exp(Δ * A)
        # We use einsum: delta (b,l,d_in) x A (d_in,n) -> (b,l,d_in,n)
        deltaA = torch.exp(torch.einsum('bld, dn -> bldn', delta, A))

        # discretise B: approximate ∫_0^Δ exp(sA) ds ≈ Δ * B.  Multiply elementwise
        # by u to scale inputs.  deltaB_u has shape (b,l,d_in,n).
        # Achieve delta * u * B broadcast over n dimension.
        # u: (b,l,d_in), B: (b,l,n)
        deltaB_u = delta.unsqueeze(-1) * u.unsqueeze(-1) * B.unsqueeze(2)

        # recurrent scan: x_state has shape (b, d_in, n)
        x_state = torch.zeros((b, d_in, n), device=u.device, dtype=u.dtype)
        outputs = []
        for t in range(l):
            # x_{t+1} = deltaA_t * x_t + deltaB_u_t
            x_state = deltaA[:, t] * x_state + deltaB_u[:, t]
            # y_t = x_t * C_t^T; C_t shape (b,n) -> broadcast along d_in
            y_t = torch.einsum('bdn, bn -> bd', x_state, C[:, t])
            outputs.append(y_t)

        # stack along time axis
        y = torch.stack(outputs, dim=1)  # (b, l, d_in)

        # add skip connection: u * D
        y = y + u * D
        return y


class MambaModel(nn.Module):
    def __init__(self, config: MambaConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        # create a list of residual blocks
        self.layers = nn.ModuleList([
            nn.ModuleList([
                MambaBlock(config),
                RMSNorm(config.d_model),
            ])
            for _ in range(config.n_layer)
        ])
        self.norm_f = RMSNorm(config.d_model)
        # output projection (tied weights)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for block, norm in self.layers:
            residual = x
            # pre‑norm
            x_norm = norm(x)
            # apply Mamba block
            x_block = block(x_norm)
            # add residual connection
            x = residual + x_block
        # final norm and projection
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits
