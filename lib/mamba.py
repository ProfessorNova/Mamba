import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mamba(nn.Module):
    """
    Minimal Mamba block that processes sequences with a causal depthwise convolution
    followed by a simple state-space update (scan). Designed for clarity, not speed.
    Args:
        d_model: hidden size (must be divisible by headdim)
        d_state: size of the SSM state per head
        d_conv: kernel size of the depthwise convolution
        headdim: dimension per head; number of heads = d_model // headdim
    """

    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4, headdim: int = 32,
                 A_init_range=(1.0, 16.0), dt_min=1e-3, dt_max=1e-1, dt_init_floor=1e-4):
        super().__init__()
        assert d_model % headdim == 0, "d_model must be divisible by headdim"
        self.d_model = d_model
        self.headdim = headdim
        self.nheads = d_model // headdim
        self.d_state = d_state
        self.d_conv = d_conv

        # projection produces: z, x, B, C, dt
        d_in_proj = 2 * d_model + 2 * d_state + self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)

        # depthwise causal conv across [x, B, C] channels
        conv_dim = d_model + 2 * d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=d_conv,
            groups=conv_dim,
            bias=True,
            padding=d_conv - 1,
        )
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # parameters for SSM
        A = torch.empty(self.nheads).uniform_(*A_init_range)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.D._no_weight_decay = True
        with torch.no_grad():
            dt = torch.exp(
                torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
            )
            dt = torch.clamp(dt, min=dt_init_floor)
            inv_sp = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_sp)
        self.dt_bias._no_weight_decay = True

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u: [B, L, d_model]
        B, L, D = u.shape
        proj = self.in_proj(u)
        z, x, B_toks, C_toks, dt_toks = torch.split(
            proj,
            [self.d_model, self.d_model, self.d_state, self.d_state, self.nheads],
            dim=-1
        )
        # causal conv + SiLU
        xBC = torch.cat([x, B_toks, C_toks], dim=-1)
        xBC = xBC.transpose(1, 2)
        xBC = self.conv1d(xBC).transpose(1, 2)
        xBC = xBC[:, :L]
        xBC = self.act(xBC)
        x, B_toks, C_toks = torch.split(xBC, [self.d_model, self.d_state, self.d_state], dim=-1)

        A = -torch.exp(self.A_log.float()).to(u.dtype)
        D_skip = self.D.to(u.dtype)
        S = u.new_zeros(B, self.nheads, self.headdim, self.d_state)
        ys = []
        for t in range(L):
            x_t = x[:, t].view(B, self.nheads, self.headdim)
            z_t = z[:, t].view(B, self.nheads, self.headdim)
            dt_t = F.softplus(dt_toks[:, t] + self.dt_bias)
            B_t = B_toks[:, t]
            C_t = C_toks[:, t]
            dA_t = torch.exp(dt_t * A)
            S = S * dA_t.view(B, self.nheads, 1, 1)
            dBx = torch.einsum('bh,bn,bhp->bhpn', dt_t, B_t, x_t)
            S = S + dBx
            y_t = torch.einsum('bhpn,bn->bhp', S, C_t) + D_skip.view(1, self.nheads, 1) * x_t
            y_t = y_t * F.silu(z_t)
            ys.append(y_t.reshape(B, self.d_model))
        y = torch.stack(ys, dim=1)
        return self.out_proj(y)


class MambaLM(nn.Module):
    """Tiny language model using an embedding layer, a Mamba block, and an LM head."""

    def __init__(self, vocab_size, d_model=256, d_state=128, d_conv=4, headdim=64, n_layers=4, pad_id=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.blocks = nn.ModuleList([Mamba(d_model, d_state, d_conv, headdim) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.lm_head(x)
