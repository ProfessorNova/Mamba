from typing import Optional, Tuple, List

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.nn import RMSNorm


def segsum(x: torch.Tensor) -> torch.Tensor:
    """
    Stable segment-sum operator used by SSD-minimal.
    Given x (..., T), returns (..., T, T) with strictly lower-triangular
    cumulative sums and -inf on the upper triangle (including above the diagonal).
    """
    T = x.size(-1)
    # Tile along a new axis then cumsum along that axis
    x_t = repeat(x, "... t -> ... t e", e=T)  # (..., T, T)
    # Zero out everything on/above the diagonal (strictly lower-triangular for running sum)
    mask_strict = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=-1)
    x_t = x_t.masked_fill(~mask_strict, 0)
    x_segsum = torch.cumsum(x_t, dim=-2)
    # Now produce 1-SS mask including the diagonal
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=0)
    return x_segsum.masked_fill(~mask, float("-inf"))


def ssd(
        X: torch.Tensor,  # (B, L, H, P)    -- already multiplied by dt outside
        A: torch.Tensor,  # (B, L, H)       -- already A * dt (discrete drift per step)
        B: torch.Tensor,  # (B, L, H, N)
        C: torch.Tensor,  # (B, L, H, N)
        block_len: int,
        initial_states: Optional[torch.Tensor] = None,  # (B, H, P, N)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    SSD-minimal with block/chunked computation. Inputs are in *discrete* form:
    - X has been pre-multiplied by dt
    - A is A * dt (so exp(segsum(A)) gives 1-SS of the discrete decay)

    Returns:
      Y: (B, L, H, P)
      final_states: (B, H, P, N)
    """
    assert X.dtype == A.dtype == B.dtype == C.dtype
    Bsz, L, H, P = X.shape
    N = B.shape[-1]
    assert L % block_len == 0, "Sequence length must be divisible by block_len"

    # Rearrange into blocks/chunks
    Xb, Ab, Bb, Cb = [rearrange(x, "b (c l) ... -> b c l ...", l=block_len) for x in (X, A, B, C)]
    # Ab: (B, C, Lb, H) -> (B, H, C, Lb)
    Ab = rearrange(Ab, "b c l h -> b h c l")

    # cumulative sums for intra-chunk decays
    A_cumsum = torch.cumsum(Ab, dim=-1)  # (B, H, C, Lb)

    # 1) Intra-chunk diagonal blocks
    Ltri = torch.exp(segsum(Ab))  # (B, H, C, Lb, Lb)
    # Einsum over chunk-length dims with B, C and inputs X
    # shapes: Cb: (B, C, Lb, H, N), Bb: (B, C, Lb, H, N), Ltri: (B, H, C, Lb, Lb), Xb: (B, C, Lb, H, P)
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", Cb, Bb, Ltri, Xb)

    # 2) Intra-chunk state accumulation (right factor / B terms)
    decay_states = torch.exp((A_cumsum[..., -1:].contiguous() - A_cumsum))  # (B, H, C, Lb)
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", Bb, decay_states, Xb)  # (B, C, H, P, N)

    # 3) Inter-chunk recurrence across chunk boundaries via 1-SS on last entries
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])  # (B, 1, H, P, N)
    else:
        initial_states = initial_states.unsqueeze(1)  # add chunk dim

    states = torch.cat([initial_states, states], dim=1)  # (B, C+1, H, P, N)

    # decay between chunks: use last element per chunk of cumulative A
    last_A = A_cumsum[..., -1]  # (B, H, C)
    decay_chunk = torch.exp(segsum(F.pad(last_A, (1, 0))))  # (B, H, C+1, C+1)
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)  # scan across chunks
    states, final_state = new_states[:, :-1], new_states[:, -1]  # (B, C, H, P, N), (B, H, P, N)

    # 4) State -> output conversion within each chunk (left factor / C terms)
    state_decay_out = torch.exp(A_cumsum)  # (B, H, C, Lb)
    Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", Cb, states, state_decay_out)

    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
    return Y, final_state


class MambaBlock(nn.Module):
    """
    PyTorch-only Mamba‑2 block approximating the official reference implementation
    (no fused/triton kernels). Uses the discrete-time SSD minimal form.

    Args:
      d_model: model dimension
      d_state: SSM state dim per head
      d_conv: depthwise conv kernel size (causal)
      expand: inner expansion factor; d_inner = expand * d_model
      headdim: head dimension (P)
      ngroups: number of groups for (B, C)
    """

    def __init__(
            self,
            d_model: int,
            d_state: int,
            d_conv: int = 4,
            expand: int = 2,
            headdim: int = 64,
            ngroups: int = 1,
    ) -> None:
        super().__init__()
        assert headdim > 0 and expand > 0

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.headdim = headdim
        self.expand = expand

        self.d_inner = self.expand * self.d_model
        assert self.d_inner % self.headdim == 0, "d_inner must be divisible by headdim"
        self.n_heads = self.d_inner // self.headdim
        self.ngroups = self.n_heads if ngroups is None else ngroups
        assert self.n_heads % self.ngroups == 0, "n_heads must be divisible by ngroups"

        # In-projection produces: [z, x, B, C, dt]
        proj_out = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.n_heads
        self.in_proj = nn.Linear(self.d_model, proj_out, bias=False)

        # Depthwise causal conv on [x|B|C] only
        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=d_conv,
            groups=conv_dim,
            bias=False,
            padding=d_conv - 1,
        )

        # Continuous-time params -> discrete via dt predicted per token
        # A = -exp(A_log) is per-head, dt = softplus(dt_branch + dt_bias)
        self.A_log = nn.Parameter(torch.log(torch.empty(self.n_heads).uniform_(1.0, 16.0)))

        with torch.no_grad():
            dt_min, dt_max, dt_floor = 1e-3, 1e-1, 1e-4
            u = torch.rand(self.n_heads)
            dt0 = torch.exp(u * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
            dt0 = torch.clamp(dt0, min=dt_floor)
            self.dt_bias = nn.Parameter(dt0 + torch.log(-torch.expm1(-dt0)))

        # D skip (per-head)
        self.D = nn.Parameter(torch.ones(self.n_heads))

        # Gated RMSNorm and output projection back to d_model
        self.norm = RMSNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(
            self,
            x: torch.Tensor,  # (B, L, d_model)
            state: Optional[torch.Tensor] = None,  # (B, H, P, N)
            block_len: int = 64,
            reset_mask: Optional[torch.Tensor] = None,  # (B, L), 1 to cut recurrence between t-1 and t
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        Bsz, L, D = x.shape
        H, P, N = self.n_heads, self.headdim, self.d_state

        x_in = x  # residual path

        zxbcdt = self.in_proj(x_in)  # (B, L, 2*Din + 2*G*N + H)
        z, xBC, dt = torch.split(
            zxbcdt,
            [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.n_heads],
            dim=-1,
        )

        # Causal depthwise conv (on x|B|C only) — conv length is L + (k-1); crop back to L
        xBC = F.silu(self.conv1d(xBC.transpose(1, 2)).transpose(1, 2))[:, :L, :]

        # Split back into x, B, C branches
        x_branch, Bv, Cv = torch.split(
            xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1
        )

        # Reshape heads
        X = rearrange(x_branch, "b l (h p) -> b l h p", p=P)  # (B, L, H, P)

        if self.ngroups == self.n_heads:
            Bv = rearrange(Bv, "b l (h n) -> b l h n", h=H)
            Cv = rearrange(Cv, "b l (h n) -> b l h n", h=H)
        else:
            repeat_factor = H // self.ngroups
            Bv = rearrange(Bv, "b l (g n) -> b l g n", g=self.ngroups).repeat_interleave(repeat_factor, dim=2)
            Cv = rearrange(Cv, "b l (g n) -> b l g n", g=self.ngroups).repeat_interleave(repeat_factor, dim=2)

        # Nonlinearities for B, C, X as in reference
        Bv = F.silu(Bv)
        Cv = F.silu(Cv)
        X = F.silu(X)

        # Per-head A and per-token dt
        A = -torch.exp(self.A_log).to(x.dtype)  # (H,)
        dt = F.softplus(dt.to(x.dtype) + self.dt_bias.to(x.dtype))  # (B, L, H)

        # Optional resets: cut recurrence between (t-1) -> t
        if reset_mask is not None:
            # Force a very negative drift at reset steps to annihilate cross-step influence
            reset_log = (-1e9) * reset_mask.unsqueeze(-1).to(dt.dtype)  # (B, L, 1)
            A_dt = (A * dt + reset_log).to(X.dtype)  # (B, L, H)
            dt_eff = F.relu(dt + reset_log)  # (B, L, H) -> zero where resets
            X_dt = (X * dt_eff.unsqueeze(-1)).to(X.dtype)  # (B, L, H, P)
        else:
            A_dt = (A * dt).to(X.dtype)
            X_dt = (X * dt.unsqueeze(-1)).to(X.dtype)

        # SSD minimal (discrete)
        Y, final_state = ssd(
            X=X_dt,
            A=A_dt,
            B=Bv,
            C=Cv,
            block_len=block_len,
            initial_states=state,
        )  # (B, L, H, P), (B, H, P, N)

        # D skip connection from input to output per head
        Y = Y + rearrange(self.D, "h -> 1 1 h 1") * X

        # Gated RMSNorm with z as the gate
        y_flat = rearrange(Y, "b l h p -> b l (h p)")  # (B, L, d_inner)
        gate = torch.sigmoid(z)  # (B, L, d_inner)
        y_core = self.norm(y_flat) * gate
        y = self.out_proj(y_core)  # (B, L, d_model)

        # Residual
        y = y + x_in
        return y, final_state


class MambaLM(nn.Module):
    """
    PyTorch-only Mamba‑2 language model (stack of MambaBlock).

    Args:
      vocab_size: vocabulary size
      d_model: model dimension
      n_layers: number of blocks
      d_state: SSM state dimension
      d_conv: depthwise conv kernel size
      expand: inner expansion factor
      headdim: per-head embedding dimension
      ngroups: grouping for B/C
    """

    def __init__(
            self,
            vocab_size: int,
            d_model: int = 256,
            n_layers: int = 4,
            d_state: int = 64,
            d_conv: int = 4,
            expand: int = 2,
            headdim: int = 128,
            ngroups: int = 1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers

        # Embedding
        self.emb = nn.Embedding(vocab_size, d_model)

        # Stack of Mamba-2 blocks
        self.layers = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    headdim=headdim,
                    expand=expand,
                    ngroups=ngroups,
                )
                for _ in range(n_layers)
            ]
        )

        # Final norm and tied output head
        self.final_norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.emb.weight

    def forward(
            self,
            tokens: torch.Tensor,  # (B, L)
            states: Optional[List[torch.Tensor]] = None,  # list of (B, H, P, N)
            reset_mask: Optional[torch.Tensor] = None,  # (B, L)
            block_len: int = 64,
            return_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        x = self.emb(tokens)  # (B, L, D)
        if states is None:
            states = [None] * self.n_layers
        new_states: List[torch.Tensor] = []

        for i, layer in enumerate(self.layers):
            x, s = layer(
                x,
                state=states[i],
                block_len=block_len,
                reset_mask=reset_mask,
            )
            new_states.append(s)

        x = self.final_norm(x)
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
            top_p: Optional[float] = None,
            eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        out_tokens = tokens.clone()
        cur_states = states
        if cur_states is None:
            # Warm up states by running the prompt once
            _, cur_states = self(
                tokens,
                states=None,
                block_len=1,
                return_state=True,
            )

        finished = torch.zeros(tokens.size(0), dtype=torch.bool, device=tokens.device)

        for _ in range(max_new_tokens):
            logits, cur_states = self(
                tokens[:, -1:],
                states=cur_states,
                block_len=1,
                return_state=True,
            )
            logits = logits[:, -1] / max(temperature, 1e-6)

            if top_k is not None:
                topk_vals, topk_idx = torch.topk(logits, top_k, dim=-1)
                masked = torch.full_like(logits, float("-inf"))
                masked.scatter_(1, topk_idx, topk_vals)
                logits = masked

            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumprobs = torch.cumsum(sorted_probs, dim=-1)
                cutoff = (cumprobs > top_p)
                cutoff[..., 1:] = cutoff[..., :-1].clone()
                cutoff[..., 0] = False
                sorted_logits[cutoff] = float("-inf")
                logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            if eos_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(-1),
                    torch.full_like(next_token, eos_id),
                    next_token,
                )

            tokens = torch.cat([tokens, next_token], dim=1)
            out_tokens = torch.cat([out_tokens, next_token], dim=1)

            if eos_id is not None:
                finished |= (next_token.squeeze(1) == eos_id)
                if torch.all(finished):
                    break

        return out_tokens
