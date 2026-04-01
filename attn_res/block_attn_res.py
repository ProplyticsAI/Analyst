"""Block Attention Residuals (AttnRes) — depth-wise softmax over block summaries.

See: Attention Residuals (arXiv:2603.15031), Eq. 2–4 and Figure 2.
"""

from __future__ import annotations

import torch


def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm over the last dimension (used as key normalization in AttnRes)."""
    r = x.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
    return x / r


def block_attn_res(
    blocks: list[torch.Tensor],
    partial_block: torch.Tensor | None,
    pseudo_query: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Inter-block attention: softmax over completed block representations plus the
    intra-block partial sum (Eq. 6). Kernel ϕ(q, k) = exp(q^T RMSNorm(k)).

    Args:
        blocks: ``[b0, ..., b_{n-1}]`` each of shape ``[B, T, D]``. ``b0`` is the
            token embedding; later entries are completed block sums.
        partial_block: Intra-block partial sum ``b_n^i`` (``[B, T, D]``), or
            ``None`` at the first sub-layer of a block (attend only over ``blocks``).
        pseudo_query: Learnable vector ``w_l`` of shape ``[D]``.
    Returns:
        Aggregated hidden state ``h`` of shape ``[B, T, D]`` (input to Attn/MLP).
    """
    if not blocks:
        raise ValueError("blocks must be non-empty (at least token embeddings b0).")

    if partial_block is None:
        v = torch.stack(blocks, dim=0)
    else:
        v = torch.stack(blocks + [partial_block], dim=0)

    # [N, B, T, D]
    k = rms_norm(v, eps=eps)
    # logits[n, b, t] = w^T RMSNorm(V)[n, b, t, :]
    logits = torch.einsum("d, n b t d -> n b t", pseudo_query, k)
    alpha = torch.softmax(logits, dim=0)
    h = torch.einsum("n b t, n b t d -> b t d", alpha, v)
    return h
