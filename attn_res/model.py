"""Minimal Transformer stack with Block Attention Residuals (Figure 2, arXiv:2603.15031)."""

from __future__ import annotations

import torch
import torch.nn as nn

from attn_res.block_attn_res import block_attn_res


class BlockAttnResTransformerLayer(nn.Module):
    """One PreNorm sub-layer pair: optional RMSNorm + attention or MLP."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_ff: int,
        dropout: float,
        is_attention: bool,
    ) -> None:
        super().__init__()
        self.is_attention = is_attention
        self.norm = nn.RMSNorm(d_model)
        if is_attention:
            self.attn = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(d_model, dim_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_ff, d_model),
                nn.Dropout(dropout),
            )
        # Pseudo-query w_l ∈ R^d; paper §5: initialize to zero (uniform attention at init).
        self.pseudo_query = nn.Parameter(torch.zeros(d_model))

    def forward(
        self,
        attn_mask: torch.Tensor | None,
        blocks: list[torch.Tensor],
        partial_block: torch.Tensor | None,
    ) -> tuple[list[torch.Tensor], torch.Tensor | None, torch.Tensor]:
        """
        Args:
            blocks: Completed block tensors ``[b0, ...]``; ``b0`` is embeddings.
            partial_block: Intra-block partial sum ``b_n^i``, or ``None`` only before
                the first sublayer right after a block boundary (Eq. 6, first row).
        Returns:
            ``(blocks_out, partial_block_out, h, residual_out)`` — ``h`` is the AttnRes
            aggregate before this sublayer; ``residual_out`` is ``norm(h) + sublayer(h)``.
        """
        pq = self.pseudo_query
        h = block_attn_res(blocks, partial_block, pq)

        z = self.norm(h)
        if self.is_attention:
            out, _ = self.attn(z, z, z, attn_mask=attn_mask, need_weights=False)
        else:
            out = self.mlp(z)

        if partial_block is None:
            partial_block = out
        else:
            partial_block = partial_block + out

        # Local residual on the AttnRes aggregate (parallel to partial_block accumulation in Figure 2).
        residual_out = z + out
        return blocks, partial_block, h, residual_out


class BlockAttnResTransformer(nn.Module):
    """
    Transformer decoder stack with Block AttnRes.

    ``pairs_per_block`` is the number of (attention, MLP) pairs per AttnRes block (paper ``S``
    when counting Transformer layers, not individual attn/MLP tensors).
    Pseudo-queries are zero-initialized per §5.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        dim_ff: int,
        n_layers: int,
        pairs_per_block: int,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if n_layers % pairs_per_block != 0:
            raise ValueError("n_layers must be divisible by pairs_per_block.")
        self.d_model = d_model
        self.n_layers = n_layers
        self.pairs_per_block = pairs_per_block
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        layers: list[BlockAttnResTransformerLayer] = []
        for _ in range(n_layers):
            layers.append(
                BlockAttnResTransformerLayer(
                    d_model, n_heads, dim_ff, dropout, is_attention=True
                )
            )
            layers.append(
                BlockAttnResTransformerLayer(
                    d_model, n_heads, dim_ff, dropout, is_attention=False
                )
            )
        self.layers = nn.ModuleList(layers)
        self.out_norm = nn.RMSNorm(d_model)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        # §5: all pseudo-query vectors must be initialized to zero.
        for layer in self.layers:
            if hasattr(layer, "pseudo_query"):
                nn.init.zeros_(layer.pseudo_query)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, t = input_ids.shape
        device = input_ids.device
        positions = torch.arange(t, device=device).unsqueeze(0).expand(b, -1)
        tok = self.token_emb(input_ids)
        pos = self.pos_emb(positions)
        hidden = tok + pos
        hidden = self.dropout(hidden)

        # b0 = token embedding (paper notation h1); stored as block 0.
        blocks: list[torch.Tensor] = [tok]
        partial_block: torch.Tensor | None = None
        sublayer_in_block = 0
        sublayers_per_block = 2 * self.pairs_per_block

        causal = torch.triu(
            torch.full((t, t), float("-inf"), device=device, dtype=hidden.dtype),
            diagonal=1,
        )

        last_residual = hidden
        for layer in self.layers:
            blocks, partial_block, _h, last_residual = layer(
                causal, blocks, partial_block
            )
            sublayer_in_block += 1
            if sublayer_in_block == sublayers_per_block:
                assert partial_block is not None
                blocks.append(partial_block)
                partial_block = None
                sublayer_in_block = 0

        return self.out_norm(last_residual)


def demo_logits(
    vocab_size: int = 32000,
    batch: int = 2,
    seq: int = 16,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Small forward pass for sanity check."""
    torch.manual_seed(seed)
    m = BlockAttnResTransformer(
        vocab_size=vocab_size,
        d_model=128,
        n_heads=4,
        dim_ff=256,
        n_layers=4,
        pairs_per_block=2,
        max_seq_len=seq,
    ).to(device)
    ids = torch.randint(0, vocab_size, (batch, seq), device=device)
    return m(ids)
