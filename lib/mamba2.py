from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from functorch.einops import rearrange
from torch.nn import RMSNorm


def segsum(x: torch.Tensor) -> torch.Tensor:
    """
    Naive segment sum calculation. exp(segsum(A)) produces a 1-SS matrix,
    which is equivalent to a scalar SSM.
    """
    T = x.size(-1)
    x_cumsum = torch.cumsum(x, dim=-1)
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def ssd(X: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
        block_len: int, initial_states: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        X: (batch, length, n_heads, d_head)
        A: (batch, length, n_heads)
        B: (batch, length, n_heads, d_state)
        C: (batch, length, n_heads, d_state)
        block_len: int - length of each chunk/block
        initial_states: (batch, n_heads, d_head, d_state) or None - initial SSM states

    Return:
        Y: (batch, length, n_heads, d_head)
        final_states: (batch, n_heads, d_head, d_state)
    """
    assert X.dtype == A.dtype == B.dtype == C.dtype
    assert X.shape[1] % block_len == 0
    # Rearrange into blocks/chunks
    X, A, B, C = [rearrange(x, "b (c l) ... -> b c l ...", l=block_len) for x in (X, A, B, C)]
    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)
    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A))
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)
    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)
    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    else:
        # Add chunk dimension
        initial_states = initial_states.unsqueeze(1)
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]
    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', C, states, state_decay_out)
    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
    return Y, final_state


class MambaBlock(nn.Module):
    """
    A single Mamba‑2 block.

    Arguments:
        d_model (int): Total model dimension (D = n_heads * head_dim).
        n_heads (int): Number of heads. The head dimension is inferred as d_model // n_heads.
        d_state (int): Dimension of the SSM state. This is independent of the head dimension.
        d_conv (int, optional): Kernel size of the depthwise convolution used for local mixing. Defaults to 4.
    """

    def __init__(
            self,
            d_model: int,
            n_heads: int,
            d_state: int,
            d_conv: int = 4,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_state = d_state
        self.d_head = d_model // n_heads
        self.d_conv = d_conv

        # Project once into all SSD pieces: X, A, B, C
        # X: (n_heads * d_head) == d_model
        # A: (n_heads)
        # B: (n_heads * d_state)
        # C: (n_heads * d_state)
        proj_out = d_model + n_heads + 2 * n_heads * d_state  # per-head B,C
        self.linear_proj = nn.Linear(d_model, proj_out, bias=False)

        # Depthwise 1D conv for local mixing (causal via manual left padding)
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=d_conv,
            groups=d_model,
            bias=False,
            padding=0,
        )

        # Gate projection
        self.gate_proj = nn.Linear(d_model, d_model, bias=False)

        # Normalization before output projection
        self.norm = RMSNorm(d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
            self,
            x: torch.Tensor,
            state: Optional[torch.Tensor] = None,
            block_len: int = 64,
            reset_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the Mamba-2 block to a sequence.

        Arguments:
            x: (batch, length, d_model) - input sequence
            state: (batch, n_heads, d_head, d_state) or None - previous SSM state
            block_len: int - chunk length for SSM computation; default 64
            reset_mask: (batch, length) or None - binary mask; 1 cuts the recurrence between t-1 and t

        Returns:
            y: (batch, length, d_model) - output sequence
            final_state: (batch, n_heads, d_head, d_state) - final SSM state
        """
        Bsz, L, D = x.shape
        H, P, N = self.n_heads, self.d_head, self.d_state

        # Local (causal) depthwise mixing + SiLU gate
        # expects (B, C, L); left-pad by (k-1) for causality.
        x_ch = x.transpose(1, 2)  # (B, D, L)
        if self.d_conv > 1:
            x_conv = self.conv1d(F.pad(x_ch, (self.d_conv - 1, 0)))  # (B, D, L)
        else:
            x_conv = self.conv1d(x_ch)  # degenerate 1x1 depthwise
        x_mixed = F.silu(x_conv.transpose(1, 2)) * x  # (B, L, D)

        # One big projection -> split into X, A, B, C
        proj = self.linear_proj(x_mixed)  # (B, L, D + H + 2*H*N)
        off0 = 0
        x_part = proj[..., off0: off0 + D]
        off0 += D  # (B, L, D)
        a_part = proj[..., off0: off0 + H]
        off0 += H  # (B, L, H)
        b_part = proj[..., off0: off0 + H * N]
        off0 += H * N  # (B, L, H*N)
        c_part = proj[..., off0: off0 + H * N]

        # Reshape / map to SSD pieces
        X = x_part.view(Bsz, L, H, P).contiguous()  # (B, L, H, P)
        A = F.logsigmoid(a_part)  # (B, L, H), ensures stability

        # Share B,C across heads -> broadcast, then apply activation
        Bv = F.silu(b_part.view(Bsz, L, H, N).contiguous())  # (B, L, H, N)
        Cv = F.silu(c_part.view(Bsz, L, H, N).contiguous())  # (B, L, H, N)
        X = F.silu(X)

        # Optional resets: break recurrence between t-1 and t where reset_mask[t] = 1
        if reset_mask is not None:
            # shape (B, L, 1) -> broadcast over H
            reset_log = (-1e9) * reset_mask.unsqueeze(-1).to(A.dtype)  # (B, L, 1)
            A = A + reset_log  # (B, L, H)

        # SSD compute
        Y, final_state = ssd(
            X=X,
            A=A,
            B=Bv,
            C=Cv,
            block_len=block_len,
            initial_states=state,
        )  # Y: (B, L, H, P)

        gate = torch.sigmoid(self.gate_proj(x_mixed))  # (B, L, D)
        y_core = Y.reshape(Bsz, L, D) * gate  # selective gating on SSM output
        y = self.out_proj(self.norm(y_core))

        # Residual connection
        y = y + x
        return y, final_state


class MambaLM(nn.Module):
    """
    A simple language model built from stacked Mamba‑2 blocks.

    This model embeds tokens into continuous representations, applies a
    sequence of Mamba‑2 blocks, normalizes the output, and projects to
    vocabulary logits. The embedding and output projection weights are tied.

    Arguments:
        vocab_size (int): Size of the vocabulary.
        d_model (int): Total model dimension (D = n_heads * head_dim). Default
        n_layers (int): Number of Mamba‑2 blocks to stack. Default is 4.
        n_heads (int): Number of attention heads. Default is 4.
        d_state (int): Dimension of the SSM state in each head. Default is
        d_conv (int): Kernel size of the depthwise convolution used for local mixing. Default is 4.
    """

    def __init__(
            self,
            vocab_size: int,
            d_model: int = 256,
            n_layers: int = 4,
            n_heads: int = 4,
            d_state: int = 64,
            d_conv: int = 4,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_state = d_state

        # Embedding layer
        self.emb = nn.Embedding(vocab_size, d_model)

        # Stack of Mamba-2 blocks
        self.layers = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_state=d_state,
                    d_conv=d_conv,
                )
                for _ in range(n_layers)
            ]
        )

        # Final normalization before logits
        self.final_norm = RMSNorm(d_model)

        # Output projection; weight tied to embedding
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.emb.weight

    def forward(
            self,
            tokens: torch.Tensor,
            states: Optional[List[torch.Tensor]] = None,
            reset_mask: Optional[torch.Tensor] = None,
            block_len: int = 64,
            return_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass of the language model.

        Arguments:
            tokens: (batch, length) - input token IDs
            states: optional list of per-layer states; each tensor is (batch, n_heads, d_head, d_state)
            reset_mask: (batch, length) or None - binary mask indicating where to reset states
            block_len: chunk length for SSM computation; default 64
            return_state: whether to return the new states

        Returns:
            logits: (batch, length, vocab_size)
            new_states: list of tensors with shape (batch, n_heads, d_head, d_state) for each layer (or None)
        """
        x = self.emb(tokens)  # (B, L, D)
        if states is None:
            states = [None] * self.n_layers
        new_states = []
        for i, layer in enumerate(self.layers):
            x, s = layer(
                x,
                state=states[i],
                block_len=block_len,
                reset_mask=reset_mask
            )
            new_states.append(s)
        x = self.final_norm(x)
        logits = self.head(x)
        return (logits, new_states) if return_state else (logits, None)

    @torch.no_grad()
    def generate(
            self,
            tokens: torch.Tensor,
            max_new_tokens: int,
            states: Optional[List[torch.Tensor]] = None,
            temperature: float = 1.0,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Arguments:
            tokens (torch.Tensor): (B, L) - input sequence of token IDs
            max_new_tokens (int): maximum number of new tokens to generate
            states: optional list of cached per-layer states, each (batch, n_heads, d_head, d_state)
            temperature (float): sampling temperature
            top_k (int, optional): if set, use top-k sampling
            top_p (float, optional): if set, use nucleus (top-p) sampling
            eos_id (int, optional): if set, stop generation when this token ID is generated

        Returns:
            out_tokens: (batch, length + max_new_tokens)
        """
        out_tokens = tokens.clone()
        cur_states = states
        if cur_states is None:
            # Initialize states by running the model on the prompt
            _, cur_states = self(
                tokens,
                states=None,
                block_len=1,
                return_state=True
            )

        finished = torch.zeros(tokens.size(0), dtype=torch.bool, device=tokens.device)

        for _ in range(max_new_tokens):
            logits, cur_states = self(
                tokens[:, -1:],
                states=cur_states,
                block_len=1,
                return_state=True
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
                # mask tokens that would exceed the cumulative probability p
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
