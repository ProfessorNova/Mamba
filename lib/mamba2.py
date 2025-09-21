from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Root mean square normalization.

    This implementation normalizes the input along the last dimension,
    multiplies by a learned weight, and avoids subtracting the mean.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., D)
        # Compute RMS along the last dimension
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * x / rms


class MambaBlock(nn.Module):
    """
    A single Mamba‑2 block implementing a selective SSM with local mixing.

    Parameters
    ----------
    d_model : int
        Total model dimension (D = n_heads * head_dim).
    n_heads : int
        Number of heads. The head dimension is inferred as d_model // n_heads.
    d_state : int
        Dimension of the SSM state. This is independent of the head dimension.
    d_conv : int, optional
        Kernel size of the depthwise convolution used for local mixing. Defaults to 4.
    dt_min, dt_max : float, optional
        Range for the learned time step dt. Delta is computed as exp(-dt).
    dropout : float, optional
        Dropout probability applied after the output projection.
    use_norm : bool, optional
        Whether to apply RMSNorm to the block input.
    """

    def __init__(
            self,
            d_model: int,
            n_heads: int,
            d_state: int,
            d_conv: int = 4,
            dt_min: float = 1e-4,
            dt_max: float = 0.1,
            dropout: float = 0.0,
            use_norm: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_state = d_state
        self.d_head = d_model // n_heads
        self.d_conv = d_conv
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dropout = dropout

        # Input normalization
        self.norm = RMSNorm(d_model) if use_norm else nn.Identity()

        # Parallel projections: dt (per head), B and C (per head and state)
        # dt_proj outputs (B, L, n_heads)
        self.dt_proj = nn.Linear(d_model, n_heads)
        # param_proj outputs (B, L, 2 * n_heads * d_state) for B and C
        self.param_proj = nn.Linear(d_model, 2 * n_heads * d_state)

        # Depthwise causal conv for local mixing
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=d_conv,
            groups=d_model,
            padding=d_conv - 1,
            bias=True,
        )

        # Gate projection to mix SSM output and input
        self.gate_proj = nn.Linear(d_model, d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        # Extra normalization before output projection (NormFormer style)
        self.extra_norm = RMSNorm(d_model)

        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)

    def forward(
            self,
            x: torch.Tensor,
            state: Optional[torch.Tensor] = None,
            reset_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the Mamba‑2 block to a sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, L, D).
        state : torch.Tensor, optional
            Previous state of shape (B, n_heads, d_state). If None, initialized to zeros.
        reset_mask : torch.Tensor, optional
            A binary mask of shape (B, L) indicating where to reset the state.

        Returns
        -------
        y : torch.Tensor
            Output tensor of shape (B, L, D).
        new_state : torch.Tensor
            New state tensor of shape (B, n_heads, d_state).
        """
        B, L, D = x.shape
        H = self.n_heads
        N = self.d_state

        # Normalize input
        x_norm = self.norm(x)

        # Compute dt_raw, B_t, C_t from the normalized input (parallel projection)
        dt_raw = self.dt_proj(x_norm)  # (B, L, H)
        param_out = self.param_proj(x_norm)  # (B, L, 2*H*N)
        param_out = param_out.view(B, L, H, 2, N)  # (B, L, H, 2, N)
        B_t, C_t = param_out.unbind(dim=3)  # each (B, L, H, N)

        # Convert dt_raw to dt in [dt_min, dt_max], then delta = exp(-dt)
        dt = self.dt_min + torch.sigmoid(dt_raw) * (self.dt_max - self.dt_min)  # (B, L, H)
        delta = torch.exp(-dt)  # (B, L, H)

        # Local mixing via depthwise causal convolution
        # Transpose to (B, D, L) for conv1d
        xt = x_norm.transpose(1, 2).contiguous()
        x_conv = F.silu(self.conv1d(xt)).transpose(1, 2)  # (B, L, D)
        # Trim padding to keep strict causality
        x_conv = x_conv[:, :L, :]

        # Gating between SSM output and input
        gate = torch.sigmoid(self.gate_proj(x_conv))  # (B, L, D)

        # Prepare initial state
        if state is None:
            h = torch.zeros(B, H, N, dtype=x.dtype, device=x.device)
        else:
            h = state

        # Perform per-step recurrence
        hs = []  # list to collect states at each timestep
        for t in range(L):
            # Reset state if reset_mask is provided
            if reset_mask is not None:
                reset = reset_mask[:, t].unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
                h = h * ~reset  # reset state where mask is 1
            # Update state: h_t = delta_t * h_{t-1} + B_t
            h = delta[:, t].unsqueeze(-1) * h + B_t[:, t]
            hs.append(h)
        # Stack to form (B, L, H, N)
        h_all = torch.stack(hs, dim=1)

        # Compute SSM output: (C_t * h_all) -> (B, L, D) via reshape
        ssm_out = (C_t * h_all).reshape(B, L, D)

        # Mix SSM output with the input using the gate
        mixed = gate * ssm_out + (1.0 - gate) * x_norm

        # Extra normalization before output projection
        mixed_norm = self.extra_norm(mixed)

        # Output projection + dropout + residual connection
        y = self.out_proj(mixed_norm)
        y = self.dropout_layer(y)
        y = y + x

        # New state is the last hidden state
        new_state = h_all[:, -1]
        return y, new_state


class MambaLM(nn.Module):
    """
    A simple language model built from stacked Mamba‑2 blocks.

    This model embeds tokens into continuous representations, applies a
    sequence of Mamba‑2 blocks, normalizes the output, and projects to
    vocabulary logits. The embedding and output projection weights are tied.
    """

    def __init__(
            self,
            vocab_size: int,
            d_model: int = 256,
            n_layers: int = 4,
            n_heads: int = 4,
            d_state: int = 64,
            dropout: float = 0.1,
            dt_min: float = 1e-4,
            dt_max: float = 0.1,
            d_conv: int = 4,
            use_norm: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_state = d_state
        self.dropout = dropout

        # Embedding layer
        self.emb = nn.Embedding(vocab_size, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        # Stack of Mamba‑2 blocks
        self.layers = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_state=d_state,
                    d_conv=d_conv,
                    dt_min=dt_min,
                    dt_max=dt_max,
                    dropout=dropout,
                    use_norm=use_norm,
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
            return_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass of the language model.

        Parameters
        ----------
        tokens : torch.Tensor
            Token IDs of shape (B, L).
        states : list of torch.Tensor, optional
            A list of previous states for each layer. Each state should have
            shape (B, n_heads, d_state). If None, all states are initialized
            to zeros.
        reset_mask : torch.Tensor, optional
            A binary mask of shape (B, L) indicating where to reset the states.
        return_state : bool, optional
            If True, returns the new states for each layer.

        Returns
        -------
        logits : torch.Tensor
            Vocabulary logits of shape (B, L, vocab_size).
        new_states : list of torch.Tensor or None
            List of new states for each layer if return_state is True; otherwise
            None.
        """
        x = self.emb(tokens)  # (B, L, D)
        x = self.emb_dropout(x)
        if states is None:
            states = [None] * self.n_layers
        new_states = []
        for i, layer in enumerate(self.layers):
            x, s = layer(x, state=states[i], reset_mask=reset_mask)
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

        Parameters
        ----------
        tokens : torch.Tensor
            Starting sequence of shape (B, L).
        max_new_tokens : int
            Maximum number of new tokens to generate.
        states : list of torch.Tensor, optional
            Cached states for each layer; see forward.
        temperature : float
            Temperature for sampling; >1.0 flattens the distribution.
        top_k : int, optional
            Top‑k sampling; if provided, only the k most likely tokens are
            considered.
        top_p : float, optional
            Nucleus (top‑p) sampling; if provided, selects the smallest set of
            tokens with cumulative probability <= p.
        eos_id : int, optional
            If provided, generation stops when all sequences emit this token.

        Returns
        -------
        tokens : torch.Tensor
            The original sequence concatenated with the generated tokens.
        """
        out_tokens = tokens.clone()
        cur_states = states
        if cur_states is None:
            # Initialize states by running the model on the prompt
            _, cur_states = self(tokens, states=None, return_state=True)

        finished = torch.zeros(tokens.size(0), dtype=torch.bool, device=tokens.device)

        for _ in range(max_new_tokens):
            logits, cur_states = self(tokens[:, -1:], states=cur_states, return_state=True)
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
