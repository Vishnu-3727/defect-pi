"""Run every module's self-check, plus the checks that span modules.

No pytest, no fixtures. Each module carries its own `demo()` with assertions,
and this file runs them all in dependency order so one command tells you
whether the system is intact.

    uv run python tests/test_pipeline.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Dependency order: a failure in config explains failures in everything after it,
# so running them in this sequence makes the first error the useful one.
MODULES = [
    "inspector.config",
    "inspector.preprocess",
    "inspector.capture",
    "inspector.decision",
    "inspector.storage",
    "inspector.actuator",
    "inspector.detector",
    "inspector.anomaly",
    "inspector.pipeline",
]


def run_module_checks() -> list[tuple[str, bool, str]]:
    import importlib

    results = []
    for name in MODULES:
        try:
            module = importlib.import_module(name)
            module.demo()
            results.append((name, True, ""))
        except Exception:
            results.append((name, False, traceback.format_exc(limit=3)))
    return results


def test_dashboard_endpoints() -> None:
    """The dashboard must serve real numbers from a live pipeline.

    Uses Flask's test client, so no port is opened and nothing hangs waiting
    for a browser.
    """
    import dataclasses
    import tempfile

    sys.path.insert(0, str(REPO_ROOT / "dashboard"))
    from app import create_app

    from inspector.config import load_config
    from inspector.pipeline import Pipeline

    cfg = load_config()
    images = cfg.resolve("bench_fixtures/images")
    if not images.is_dir():
        images = cfg.resolve("data/yolo/test/images")

    with tempfile.TemporaryDirectory() as tmp:
        cfg = dataclasses.replace(
            cfg,
            storage=dataclasses.replace(cfg.storage, db=str(Path(tmp) / "d.db"),
                                        image_dir=str(Path(tmp) / "fail")),
            actuator=dataclasses.replace(cfg.actuator, backend="mock"),
        )
        with Pipeline(cfg, source_override=str(images)) as pipe:
            pipe.run(max_frames=12)
            client = create_app(pipe).test_client()

            assert client.get("/").status_code == 200

            stats = client.get("/api/stats").get_json()
            assert stats["live"]["frames"] == 12, stats["live"]
            assert stats["database"]["total"] == 12, stats["database"]
            assert stats["config"]["actuator"] == "mock"

            recent = client.get("/api/recent").get_json()
            assert len(recent) == 12, len(recent)
            assert {"verdict", "reason", "latency_ms"} <= set(recent[0])

            # The stream must publish an actual JPEG, not an empty response.
            frame = pipe.latest_jpeg()
            assert frame and frame.startswith(b"\xff\xd8"), "not a JPEG"


def test_runtime_has_no_training_dependency() -> None:
    """Nothing importable from `inspector` may pull in torch.

    This is the invariant that makes the Pi deployment possible at all, and it
    is easy to break by adding one convenient import. Checking it here means
    the break is caught immediately rather than on the Pi.
    """
    import importlib
    import inspect as stdlib_inspect

    banned = {"torch", "ultralytics", "torchvision"}
    for name in MODULES:
        module = importlib.import_module(name)
        source = stdlib_inspect.getsource(module)
        for word in banned:
            assert f"import {word}" not in source, \
                f"{name} imports {word}; the Pi runtime must not need it"


def main() -> int:
    print("module self-checks")
    print("=" * 62)
    results = run_module_checks()

    print()
    print("cross-module checks")
    print("=" * 62)
    for fn in (test_dashboard_endpoints, test_runtime_has_no_training_dependency):
        try:
            fn()
            results.append((fn.__name__, True, ""))
            print(f"  {fn.__name__}: passed")
        except Exception:
            results.append((fn.__name__, False, traceback.format_exc(limit=3)))
            print(f"  {fn.__name__}: FAILED")

    failed = [(n, t) for n, ok, t in results if not ok]
    print()
    print("=" * 62)
    print(f"{len(results) - len(failed)}/{len(results)} passed")

    for name, trace in failed:
        print()
        print(f"--- {name}")
        print(trace)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
