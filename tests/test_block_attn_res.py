import torch

from attn_res.block_attn_res import block_attn_res, rms_norm
from attn_res.model import BlockAttnResTransformer


def test_block_attn_res_uniform_when_query_zero():
    torch.manual_seed(0)
    b, t, d = 2, 4, 8
    n = 3
    blocks = [torch.randn(b, t, d) for _ in range(n)]
    partial = torch.randn(b, t, d)
    w = torch.zeros(d)
    h = block_attn_res(blocks, partial, w)
    v = torch.stack(blocks + [partial], dim=0)
    expected = v.mean(dim=0)
    assert torch.allclose(h, expected, atol=1e-5)


def test_rms_norm_matches_manual():
    x = torch.tensor([[3.0, 4.0]])
    y = rms_norm(x)
    r = (x.pow(2).mean(-1, keepdim=True).sqrt())
    assert torch.allclose(y, x / r)


def test_transformer_forward_shape():
    m = BlockAttnResTransformer(
        vocab_size=100,
        d_model=32,
        n_heads=4,
        dim_ff=64,
        n_layers=4,
        pairs_per_block=2,
        max_seq_len=32,
    )
    ids = torch.randint(0, 100, (2, 16))
    y = m(ids)
    assert y.shape == (2, 16, 32)


def test_gradient_flow():
    m = BlockAttnResTransformer(
        vocab_size=50,
        d_model=16,
        n_heads=2,
        dim_ff=32,
        n_layers=2,
        pairs_per_block=1,
        max_seq_len=16,
    )
    ids = torch.randint(0, 50, (1, 8))
    y = m(ids)
    y.sum().backward()
    assert m.layers[0].pseudo_query.grad is not None
    assert not torch.isnan(m.layers[0].pseudo_query.grad).any()
