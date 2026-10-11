"""YAML config -> validated, frozen dataclasses.

Why not read the YAML dict directly everywhere? Because a typo in a key then
surfaces as a KeyError deep inside the inference loop, three hours into a demo.
Validating once at startup turns every config mistake into one clear message
before a single frame is captured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CaptureConfig:
    source: str
    device_index: int
    path: str
    width: int
    height: int
    fps: int
    replay_fps: float


@dataclass(frozen=True)
class DetectorConfig:
    model: str
    imgsz: int
    conf_threshold: float
    iou_threshold: float
    num_threads: int


@dataclass(frozen=True)
class AnomalyConfig:
    enabled: bool
    bank: str
    imgsz: int
    gate_conf_below: float
    max_calls_per_second: float
    score_threshold: float


@dataclass(frozen=True)
class DecisionConfig:
    min_defect_area_px: int


@dataclass(frozen=True)
class StorageConfig:
    db: str
    save_fail_images: bool
    image_dir: str


@dataclass(frozen=True)
class ActuatorConfig:
    backend: str
    pulse_ms: int
    port: str
    baud: int
    pin: int
    eject_delay_s: float


@dataclass(frozen=True)
class DashboardConfig:
    host: str
    port: int
    jpeg_quality: int


@dataclass(frozen=True)
class Config:
    classes: list[str]
    capture: CaptureConfig
    detector: DetectorConfig
    anomaly: AnomalyConfig
    decision: DecisionConfig
    storage: StorageConfig
    actuator: ActuatorConfig
    dashboard: DashboardConfig
    source_path: Path = field(default=REPO_ROOT / "configs" / "default.yaml")

    def resolve(self, relative: str) -> Path:
        """Turn a config-relative path into an absolute one.

        Paths in the YAML are relative to the repo root so the same file works
        on Windows and on the Pi without editing.
        """
        p = Path(relative)
        return p if p.is_absolute() else REPO_ROOT / p


VALID_CAPTURE_SOURCES = {"auto", "picamera", "device", "video", "folder"}
VALID_ACTUATOR_BACKENDS = {"mock", "serial", "gpio"}


def _section(raw: dict, name: str) -> dict:
    if name not in raw:
        raise ValueError(f"config is missing the '{name}' section")
    if not isinstance(raw[name], dict):
        raise ValueError(f"config section '{name}' must be a mapping")
    return raw[name]


def _build(cls, raw: dict, name: str):
    """Instantiate a dataclass from a config section, naming every bad key."""
    expected = {f.name for f in cls.__dataclass_fields__.values()}
    got = set(raw)
    unknown = got - expected
    missing = expected - got
    if unknown:
        raise ValueError(f"config section '{name}' has unknown keys: {sorted(unknown)}")
    if missing:
        raise ValueError(f"config section '{name}' is missing keys: {sorted(missing)}")
    return cls(**raw)


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else REPO_ROOT / "configs" / "default.yaml"
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping, got {type(raw).__name__}")

    classes = raw.get("classes")
    if not isinstance(classes, list) or not classes:
        raise ValueError("config 'classes' must be a non-empty list")

    cfg = Config(
        classes=list(classes),
        capture=_build(CaptureConfig, _section(raw, "capture"), "capture"),
        detector=_build(DetectorConfig, _section(raw, "detector"), "detector"),
        anomaly=_build(AnomalyConfig, _section(raw, "anomaly"), "anomaly"),
        decision=_build(DecisionConfig, _section(raw, "decision"), "decision"),
        storage=_build(StorageConfig, _section(raw, "storage"), "storage"),
        actuator=_build(ActuatorConfig, _section(raw, "actuator"), "actuator"),
        dashboard=_build(DashboardConfig, _section(raw, "dashboard"), "dashboard"),
        source_path=path,
    )
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    if cfg.capture.source not in VALID_CAPTURE_SOURCES:
        raise ValueError(
            f"capture.source must be one of {sorted(VALID_CAPTURE_SOURCES)}, "
            f"got '{cfg.capture.source}'"
        )
    if cfg.capture.source in {"video", "folder"} and not cfg.capture.path:
        raise ValueError(f"capture.path is required when source is '{cfg.capture.source}'")

    if cfg.actuator.backend not in VALID_ACTUATOR_BACKENDS:
        raise ValueError(
            f"actuator.backend must be one of {sorted(VALID_ACTUATOR_BACKENDS)}, "
            f"got '{cfg.actuator.backend}'"
        )

    if not 0.0 < cfg.capture.replay_fps <= 120.0:
        raise ValueError("capture.replay_fps must be in (0, 120]")
    if not 0.0 < cfg.detector.conf_threshold < 1.0:
        raise ValueError("detector.conf_threshold must be strictly between 0 and 1")
    if not 0.0 < cfg.detector.iou_threshold < 1.0:
        raise ValueError("detector.iou_threshold must be strictly between 0 and 1")
    if cfg.detector.imgsz % 32 != 0:
        raise ValueError(
            f"detector.imgsz must be a multiple of 32 (the network stride), "
            f"got {cfg.detector.imgsz}"
        )
    if cfg.detector.num_threads < 1:
        raise ValueError("detector.num_threads must be at least 1")

    if cfg.anomaly.enabled:
        if cfg.anomaly.imgsz % 32 != 0:
            raise ValueError("anomaly.imgsz must be a multiple of 32")
        if cfg.anomaly.max_calls_per_second <= 0:
            raise ValueError("anomaly.max_calls_per_second must be positive")

    if not 1 <= cfg.dashboard.jpeg_quality <= 100:
        raise ValueError("dashboard.jpeg_quality must be between 1 and 100")
    if cfg.actuator.eject_delay_s < 0:
        raise ValueError("actuator.eject_delay_s cannot be negative")


def demo() -> None:
    """Self-check: the shipped config loads, and bad values are rejected."""
    cfg = load_config()
    assert len(cfg.classes) == 6, cfg.classes
    assert cfg.detector.imgsz % 32 == 0
    assert cfg.actuator.backend in VALID_ACTUATOR_BACKENDS
    assert cfg.resolve("models/x.onnx").is_absolute()

    import copy

    raw = yaml.safe_load((REPO_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))

    bad = copy.deepcopy(raw)
    bad["detector"]["imgsz"] = 300  # not a multiple of 32
    tmp = REPO_ROOT / "configs" / "_selfcheck.yaml"
    tmp.write_text(yaml.safe_dump(bad), encoding="utf-8")
    try:
        load_config(tmp)
        raise AssertionError("imgsz=300 should have been rejected")
    except ValueError as e:
        assert "multiple of 32" in str(e), e
    finally:
        tmp.unlink()

    print(f"config OK: {len(cfg.classes)} classes, imgsz={cfg.detector.imgsz}, "
          f"actuator={cfg.actuator.backend}")


if __name__ == "__main__":
    demo()
