"""The headline claim, as a test: one weight set, any sensor count.

If this ever fails, the project title is wrong.
"""
import torch

from unify.models.unify_rul import UnifyRUL

L, D = 30, 64


def make():
    torch.manual_seed(0)
    return UnifyRUL(d_model=D, n_heads=4, n_queries=1, sensor_vocab=64, n_domains=8).eval()


def feed(model, B, S):
    x = torch.randn(B, S, L)
    mask = torch.ones(B, S, dtype=torch.bool)
    ids = torch.arange(1, S + 1).expand(B, S)
    dom = torch.zeros(B, dtype=torch.long)
    with torch.no_grad():
        return model(x, mask, ids, dom)


def test_any_sensor_count_same_output_shape():
    m = make()
    for S in (2, 3, 10, 14, 16, 21):
        rul, attn = feed(m, 4, S)
        assert rul.shape == (4, 1), f"S={S} gave {rul.shape}"
        assert attn.shape == (4, 4, 1, S)


def test_no_weight_shape_depends_on_sensor_count():
    """The reshape folds S into the batch dim, so parameter count is S-invariant."""
    m = make()
    before = m.n_params()
    feed(m, 2, 2)
    feed(m, 2, 21)
    assert m.n_params() == before


def test_parameter_budget():
    """Sized to stay inside the Pi's 1 MB shared L2 cache."""
    n = make().n_params()
    assert 20_000 < n < 200_000, f"{n} params — check d_model"
    assert n * 4 / 1024 < 500, "fp32 weights exceed 500 KB; it will fall out of L2"


def test_mask_blocks_padded_sensors():
    """Garbage in a masked channel must not reach the output.

    Without this, the model attends to zero-padding and validation loss starts
    depending on batch composition — a baffling bug if you don't know to look.
    """
    m = make()
    x = torch.randn(1, 6, L)
    ids = torch.arange(1, 7).unsqueeze(0)
    dom = torch.zeros(1, dtype=torch.long)

    mask = torch.ones(1, 6, dtype=torch.bool)
    mask[0, -2:] = False

    with torch.no_grad():
        a, _ = m(x, mask, ids, dom)
        x2 = x.clone()
        x2[0, -2:] = 1e3                     # garbage where the mask says "ignore"
        b, _ = m(x2, mask, ids, dom)

    assert torch.allclose(a, b, atol=1e-5), "masked sensors leaked into the prediction"


def test_attention_weights_sum_to_one_over_valid_sensors():
    m = make()
    mask = torch.ones(2, 8, dtype=torch.bool)
    mask[0, 5:] = False
    with torch.no_grad():
        _, attn = m(torch.randn(2, 8, L), mask,
                    torch.arange(1, 9).expand(2, 8), torch.zeros(2, dtype=torch.long))
    valid = (attn * mask[:, None, None, :]).sum(-1)
    assert torch.allclose(valid, torch.ones_like(valid), atol=1e-4)
    # ...and essentially nothing lands on the padded positions
    assert (attn * ~mask[:, None, None, :]).sum() < 1e-3


def test_window_length_is_flexible_too():
    """Global average pooling removes the L dependence, so the same weights accept
    a different window length in PyTorch. The ONNX export pins L on purpose —
    see unify/models/encoder.py — so do not claim this end to end."""
    m = make()
    for length in (20, 30, 50):
        x = torch.randn(2, 5, length)
        with torch.no_grad():
            rul, _ = m(x, torch.ones(2, 5, dtype=torch.bool),
                       torch.arange(1, 6).expand(2, 5), torch.zeros(2, dtype=torch.long))
        assert rul.shape == (2, 1)
