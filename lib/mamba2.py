from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * x / rms


class MambaBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_heads: int,
            d_state: int,
            use_norm: bool = True,
            dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert d_state == d_model // n_heads, "d_state should equal head dim (d_model // n_heads)"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_state = d_state
        self.dropout = dropout
        self.kernel_size = 4  # small, efficient causal kernel

        self.norm = RMSNorm(d_model) if use_norm else nn.Identity()

        # Depthwise, causal Conv1d. We may run it via a fused kernel in forward.
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=self.kernel_size,
            groups=d_model,
            padding=self.kernel_size - 1,  # causal padding; we'll trim back to L
            bias=True,
        )

        # Parameter & gating projections
        self.param_proj = nn.Linear(d_model, 3 * n_heads * d_state)  # -> (delta, B, C)
        self.gate_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout_layer = nn.Dropout(dropout)

    def forward(
            self,
            x: torch.Tensor,  # (B, L, D)
            state: Optional[torch.Tensor] = None,  # (B, H, S)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        H, S = self.n_heads, self.d_state

        # Pre-norm
        x_norm = self.norm(x)  # (B, L, D)

        # Temporal mixing via causal depthwise Conv1d (fast path if available)
        xt = x_norm.transpose(1, 2).contiguous()  # (B, D, L)
        if (causal_conv1d_fn is not None) and xt.is_cuda:
            w = self.conv1d.weight.view(D, self.kernel_size).contiguous()  # (D, K)
            b = self.conv1d.bias
            x_conv = causal_conv1d_fn(x=xt, weight=w, bias=b, activation="silu").transpose(1, 2)
        else:
            x_conv = F.silu(self.conv1d(xt)).transpose(1, 2)
            x_conv = x_conv[:, :L, :]  # trim padding to keep strict causality

        # Gating and SSM parameters from the convolved features
        gate = torch.sigmoid(self.gate_proj(x_conv))  # (B, L, D)
        params = self.param_proj(x_conv).view(B, L, H, 3, S)  # (B, L, H, 3, S)
        delta, B_t, C_t = params.unbind(dim=3)  # each (B, L, H, S)
        delta = torch.sigmoid(delta)

        # Vectorized recurrence
        # h_t = prod_{i<=t} delta_i * (h_0 + sum_{i<=t} B_i / prod_{j<=i} delta_j)
        if state is None:
            state = torch.zeros(B, H, S, dtype=x.dtype, device=x.device)

        # Do the scan in float32 for stability, cast back after.
        delta32 = delta.to(torch.float32)  # (B, L, H, S)
        B32 = B_t.to(torch.float32)  # (B, L, H, S)

        P = torch.cumprod(delta32, dim=1)  # (B, L, H, S)  cumulative product
        U = B32 / (P + 1e-8)  # (B, L, H, S)
        CU = torch.cumsum(U, dim=1)  # (B, L, H, S)

        h0 = state.to(torch.float32).unsqueeze(1)  # (B, 1, H, S)
        h_all = (P * (h0 + CU)).to(x.dtype)  # (B, L, H, S)  post-update states per step

        # SSM output and mixing
        ssm = (C_t * h_all).reshape(B, L, D)  # (B, L, D)
        mixed = gate * ssm + (1.0 - gate) * x_norm  # (B, L, D)

        y = self.out_proj(mixed)
        y = self.dropout_layer(y)
        y = y + x  # residual

        new_state = h_all[:, -1]  # (B, H, S)
        return y, new_state


class MambaLM(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            d_model: int = 256,
            n_layers: int = 4,
            n_heads: int = 4,
            d_state: int = 64,
            dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_state = d_state
        self.dropout = dropout

        self.emb = nn.Embedding(vocab_size, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_state=d_state,
                    use_norm=True,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
            self,
            tokens: torch.Tensor,  # (B, L)
            states: Optional[List[torch.Tensor]] = None,  # list of (B, H, S)
            return_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        x = self.emb(tokens)
        x = self.emb_dropout(x)
        if states is None: states = [None] * self.n_layers
        new_states = []
        for i, layer in enumerate(self.layers):
            x, s = layer(x, state=states[i])
            new_states.append(s)
        x = self.norm(x)
        logits = self.head(x)
        return (logits, new_states) if return_state else (logits, None)

    @torch.no_grad()
    def generate(
            self,
            tokens: torch.Tensor,  # (B, L)
            max_new_tokens: int,
            states: Optional[List[torch.Tensor]] = None,
            temperature: float = 1.0,
            top_k: Optional[int] = None,
    ) -> torch.Tensor:
        out_tokens = tokens.clone()
        cur_states = states

        if cur_states is None:
            _, cur_states = self(tokens, states=None, return_state=True)

        for _ in range(max_new_tokens):
            logits, cur_states = self(tokens[:, -1:], states=cur_states, return_state=True)
            logits = logits[:, -1] / max(temperature, 1e-6)  # (B, V)

            if top_k is not None:
                topk_vals, topk_idx = torch.topk(logits, top_k, dim=-1)
                masked = torch.full_like(logits, float("-inf"))
                masked.scatter_(1, topk_idx, topk_vals)
                logits = masked

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            tokens = torch.cat([tokens, next_token], dim=1)
            out_tokens = torch.cat([out_tokens, next_token], dim=1)

        return out_tokens
