from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from causal_conv1d import causal_conv1d_fn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        # Compute the root mean square across the last dimension
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
        assert d_state == d_model // n_heads, (
            "For simplicity we set the state dimension equal to the head dimension."
        )
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_state = d_state
        self.dropout = dropout
        self.kernel_size = 4  # Kernel for causal convolution

        # Optional normalization at the beginning of the block to stabilize
        # training, as suggested in NormFormer and adopted in Mamba‑2【321109661959848†L1947-L1955】.
        self.norm = RMSNorm(d_model) if use_norm else nn.Identity()

        # Causal convolution to mix information from neighboring tokens.
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=self.kernel_size,
            groups=d_model,
            padding=self.kernel_size - 1,
            bias=True,
        )

        # Linear projections to compute gating and SSM parameters. We project
        # the normalized input into three sets of parameters per head:
        # * delta: (batch, n_heads, d_state)
        # * B:     (batch, n_heads, d_state)
        # * C:     (batch, n_heads, d_state)
        # We pack these into a single large projection for efficiency.
        self.param_proj = nn.Linear(d_model, 3 * n_heads * d_state)

        # Gating projection (batch, d_model). We share the gating across
        # heads but keep per‑dimension gating to allow fine‑grained control.
        self.gate_proj = nn.Linear(d_model, d_model)

        # Output projection back to d_model. This mixes the per‑head
        # contributions into the full model dimension.
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout_layer = nn.Dropout(dropout)

    def forward(
            self,
            x: torch.Tensor,
            state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seqlen, _ = x.shape
        # Normalize input
        x_norm = self.norm(x)

        # Temporal mixing via causal convolution
        bsz, seqlen, d_model = x_norm.shape
        xt = x_norm.transpose(1, 2).contiguous()  # (B, D, L)
        w = self.conv1d.weight.view(d_model, self.kernel_size).contiguous()
        bt = self.conv1d.bias
        # Using the custom CUDA kernel for efficiency
        x_conv_t = causal_conv1d_fn(x=xt, weight=w, bias=bt, activation="silu")
        x_conv = x_conv_t.transpose(1, 2)

        # Gating and SSM parameter projection
        gate = torch.sigmoid(self.gate_proj(x_conv))  # (B, L, D)

        # Compute SSM parameters (batch, seq_len, n_heads, d_state)
        params = self.param_proj(x_norm)
        params = params.view(bsz, seqlen, self.n_heads, 3, self.d_state)
        delta, B, C = params.unbind(dim=3)
        # Stabilize decay by squashing to (0, 1). Using sigmoid ensures
        # positive decay less than one, which encourages forgetting.
        delta = torch.sigmoid(delta)

        # If no initial state provided, start with zeros
        if state is None:
            state = torch.zeros(bsz, self.n_heads, self.d_state, dtype=x.dtype, device=x.device)

        outputs: List[torch.Tensor] = []
        h = state  # (batch, n_heads, d_state)

        # Process the sequence token by token. Although PyTorch supports
        # vectorized operations over the sequence dimension, implementing
        # the recurrent update in a for loop provides clarity and keeps
        # GPU memory usage low. The overhead is acceptable for moderate
        # sequence lengths (e.g. a few hundred tokens).
        for t in range(seqlen):
            # Slice parameters at time step t
            delta_t = delta[:, t]  # (batch, n_heads, d_state)
            B_t = B[:, t]  # (batch, n_heads, d_state)
            C_t = C[:, t]  # (batch, n_heads, d_state)
            gate_t = gate[:, t]  # (batch, d_model)

            # Update hidden state: h = delta_t * h + B_t
            h = delta_t * h + B_t

            # Compute SSM output: elementwise product of C_t and h
            # Shape: (batch, n_heads, d_state)
            ssm_out = C_t * h

            # Reshape ssm_out to (batch, d_model) by concatenating heads
            ssm_out = ssm_out.view(bsz, -1)

            # Mix SSM output with the original input via gating
            # We cast gate_t to the same shape (batch, d_model)
            mixed = gate_t * ssm_out + (1.0 - gate_t) * x_norm[:, t]

            # Project back to d_model and add residual
            out_t = self.out_proj(mixed)
            out_t = self.dropout_layer(out_t)
            out_t = out_t + x[:, t]

            outputs.append(out_t)

        # Stack outputs along the sequence dimension
        y = torch.stack(outputs, dim=1)
        new_state = h  # final hidden state
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

        # Token embedding and dropout
        self.emb = nn.Embedding(vocab_size, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        # Stack of MambaBlocks
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

        # Final normalization before projecting to logits
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
            self,
            tokens: torch.Tensor,
            states: Optional[List[torch.Tensor]] = None,
            return_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        # Embed tokens
        x = self.emb(tokens)  # (batch, seq_len, d_model)
        x = self.emb_dropout(x)

        # Prepare initial states for each layer
        if states is None:
            states = [None] * self.n_layers
        new_states: List[torch.Tensor] = []

        # Pass through blocks
        for i, layer in enumerate(self.layers):
            x, new_state = layer(x, state=states[i])
            new_states.append(new_state)

        # Final normalization and projection
        x = self.norm(x)
        logits = self.head(x)

        if return_state:
            return logits, new_states
        return logits, None

    @torch.no_grad()
    def generate(
            self,
            tokens: torch.Tensor,
            max_new_tokens: int,
            states: Optional[List[torch.Tensor]] = None,
            temperature: float = 1.0,
            top_k: Optional[int] = None,
    ) -> torch.Tensor:
        out_tokens = tokens.clone()
        cur_states = states

        if cur_states is None:
            logits, cur_states = self(tokens, states=None, return_state=True)

        for _ in range(max_new_tokens):
            # Only feed the last token to the model to save computation
            logits, cur_states = self(tokens[:, -1:], states=cur_states, return_state=True)
            logits = logits[:, -1] / max(temperature, 1e-6)  # (batch, vocab_size)

            if top_k is not None:
                # Mask out all but the top_k tokens
                topk_vals, topk_idx = torch.topk(logits, top_k, dim=-1)
                masked = torch.full_like(logits, float('-inf'))
                masked.scatter_(1, topk_idx, topk_vals)
                logits = masked

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)

            tokens = torch.cat([tokens, next_token], dim=1)
            out_tokens = torch.cat([out_tokens, next_token], dim=1)

        return out_tokens
