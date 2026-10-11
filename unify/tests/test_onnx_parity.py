"""Export fidelity — the check that stops you debugging from the Pi.

An export that silently changes numerics (train-mode BatchNorm, an opset
mismatch, an op quietly approximated) costs hours to diagnose on device and two
seconds to catch here.
"""
import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from unify.deploy.export_onnx import export
from unify.models.unify_rul import UnifyRUL

L = 30
CFG = {
    "data": {"window": L},
    "model": {"d_model": 32, "n_heads": 4, "n_queries": 1, "sensor_vocab": 64,
              "n_domains": 8, "dropout": 0.2, "attn_dropout": 0.1,
              "sigmoid_output": False},
    "loss": {"kind": "smooth_l1", "quantiles": [0.1, 0.5, 0.9]},
}


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    torch.manual_seed(0)
    m = UnifyRUL(d_model=32, n_heads=4, n_queries=1, sensor_vocab=64, n_domains=8)
    # Move the BatchNorm running stats away from their init so a train-mode export
    # would give visibly different numbers — otherwise this test cannot see the bug
    # it exists to catch.
    m.train()
    for _ in range(5):
        m(torch.randn(16, 9, L), torch.ones(16, 9, dtype=torch.bool),
          torch.arange(1, 10).expand(16, 9), torch.zeros(16, dtype=torch.long))

    d = tmp_path_factory.mktemp("artifacts")
    ckpt = d / "best.pt"
    torch.save({"model": m.state_dict(), "cfg": CFG, "scales": {"synthetic": 100.0},
                "norm": {}, "epoch": 0, "val": {}}, ckpt)
    path = export(ckpt, d, dummy_sensors=14, window=L)
    return path, m.eval()


def feeds(S, B=1, seed=0):
    rng = np.random.default_rng(seed)
    return {"x": rng.standard_normal((B, S, L)).astype(np.float32),
            "mask": np.ones((B, S), dtype=bool),
            "sensor_ids": np.tile(np.arange(1, S + 1, dtype=np.int64), (B, 1)),
            "domain_id": np.zeros(B, dtype=np.int64)}


def test_graph_is_structurally_valid(artifacts):
    onnx.checker.check_model(onnx.load(artifacts[0]))


def test_batch_axis_is_dynamic_in_the_declared_output_shape(artifacts):
    """Not just "it happens to work" — the graph must DECLARE the axis dynamic, or
    ORT logs a shape mismatch on every batched call and quantisation rejects it."""
    g = onnx.load(artifacts[0]).graph
    dim0 = g.output[0].type.tensor_type.shape.dim[0]
    assert dim0.dim_param == "batch", f"output batch axis is static: {dim0}"


@pytest.mark.parametrize("S", [2, 10, 14, 21])
def test_onnx_matches_pytorch_at_every_sensor_count(artifacts, S):
    path, model = artifacts
    f = feeds(S, seed=S)
    got = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(["rul"], f)[0]
    with torch.no_grad():
        want, _ = model(*[torch.from_numpy(f[k]) for k in
                          ("x", "mask", "sensor_ids", "domain_id")])
    np.testing.assert_allclose(got, want.numpy(), rtol=1e-3, atol=1e-4)


def test_batching_still_works_after_export(artifacts):
    """The Pi predicts with B=1; training used B=256. Both must be valid."""
    out = ort.InferenceSession(str(artifacts[0]), providers=["CPUExecutionProvider"]) \
        .run(["rul"], feeds(14, B=8, seed=1))[0]
    assert out.shape == (8, 1)


def test_masked_channel_cannot_inject_garbage(artifacts):
    """A sensor can fail in the field and the model keeps working — no retraining.
    That is a genuine selling point, so it gets a test."""
    sess = ort.InferenceSession(str(artifacts[0]), providers=["CPUExecutionProvider"])
    f = feeds(14, seed=7)
    f["mask"][0, -1] = False
    f["x"][0, -1] = 1e6
    out = sess.run(["rul"], f)[0]
    assert np.isfinite(out).all() and abs(float(out.ravel()[0])) < 1e3


def test_int8_quantisation_preserves_predictions(artifacts, tmp_path):
    """~2.6x smaller. Always report an accuracy number beside a size claim."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    q = tmp_path / "int8.onnx"
    quantize_dynamic(str(artifacts[0]), str(q), weight_type=QuantType.QInt8)
    assert q.stat().st_size < artifacts[0].stat().st_size

    f = feeds(14, seed=3)
    a = ort.InferenceSession(str(artifacts[0]), providers=["CPUExecutionProvider"]).run(["rul"], f)[0]
    b = ort.InferenceSession(str(q), providers=["CPUExecutionProvider"]).run(["rul"], f)[0]
    delta = float(np.abs(a - b).max())
    assert delta < 0.05, f"int8 changed the normalised prediction by {delta:.3f} of full life"
