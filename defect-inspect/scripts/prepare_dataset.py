"""Convert NEU-DET from Pascal VOC XML to YOLO format, with a stratified split.

Two formats, two conventions:

  VOC  : absolute pixel corners      <xmin> <ymin> <xmax> <ymax>
  YOLO : normalised centre + size    cls  cx/W  cy/H  w/W  h/H

Normalising by image size is what lets one model train on mixed resolutions,
and it is why a YOLO label file carries no image dimensions at all.

The split is stratified: each class is shuffled and cut 70/20/10 separately, so
every split holds the same class mix. A plain random split of 1770 images would
leave the rarer classes lumpy in val, and the mAP would then measure luck.

Run:  uv run scripts/prepare_dataset.py
"""

from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")


def _safe_parse(xml_path: Path) -> ET.Element:
    """Parse an annotation, refusing any file that declares a DTD or entities.

    These XML files come from a third-party GitHub repo, so they are untrusted
    input. Python's expat parser expands internal entities, which is the
    billion-laughs denial of service. VOC annotations never need a DTD, so
    rejecting the construct outright is both safer and cheaper than pulling in
    a hardened XML library.
    """
    text = xml_path.read_text(encoding="utf-8", errors="strict")
    lowered = text.lower()
    if "<!doctype" in lowered or "<!entity" in lowered:
        raise ValueError(f"{xml_path.name}: DTD or entity declaration is not allowed")
    return ET.fromstring(text)


def parse_voc(xml_path: Path) -> tuple[int, int, list[tuple[str, float, float, float, float]]]:
    """Return (width, height, [(class_name, xmin, ymin, xmax, ymax), ...])."""
    root = _safe_parse(xml_path)
    size = root.find("size")
    if size is None:
        raise ValueError(f"{xml_path.name}: no <size> element")
    width = int(size.findtext("width", "0"))
    height = int(size.findtext("height", "0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"{xml_path.name}: bad image size {width}x{height}")

    boxes = []
    for obj in root.iter("object"):
        name = (obj.findtext("name") or "").strip()
        bnd = obj.find("bndbox")
        if not name or bnd is None:
            continue
        boxes.append((
            name,
            float(bnd.findtext("xmin", "0")),
            float(bnd.findtext("ymin", "0")),
            float(bnd.findtext("xmax", "0")),
            float(bnd.findtext("ymax", "0")),
        ))
    return width, height, boxes


def to_yolo(box: tuple[float, float, float, float], width: int, height: int):
    """VOC corners -> YOLO normalised centre/size. Returns None if degenerate.

    NEU-DET annotations occasionally run a pixel or two past the image edge, so
    clamp before converting. Without the clamp the normalised value exceeds 1.0
    and ultralytics rejects the whole label file at load time.
    """
    xmin, ymin, xmax, ymax = box
    xmin, xmax = sorted((xmin, xmax))
    ymin, ymax = sorted((ymin, ymax))
    xmin = max(0.0, min(xmin, width))
    xmax = max(0.0, min(xmax, width))
    ymin = max(0.0, min(ymin, height))
    ymax = max(0.0, min(ymax, height))

    bw, bh = xmax - xmin, ymax - ymin
    if bw < 1.0 or bh < 1.0:
        return None
    return (
        (xmin + bw / 2) / width,
        (ymin + bh / 2) / height,
        bw / width,
        bh / height,
    )


def stratified_split(stems: list[str], ratios: tuple[float, float, float], seed: int):
    """Split per class so every split keeps the same class mix."""
    by_class: dict[str, list[str]] = defaultdict(list)
    for stem in stems:
        by_class[stem.rsplit("_", 1)[0]].append(stem)

    rng = random.Random(seed)
    out: dict[str, list[str]] = {s: [] for s in SPLITS}
    for cls in sorted(by_class):
        items = sorted(by_class[cls])
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        out["train"] += items[:n_train]
        out["val"] += items[n_train:n_train + n_val]
        out["test"] += items[n_train + n_val:]
    return out


def convert(src: Path, dst: Path, ratios, seed: int) -> dict:
    images_dir, annots_dir = src / "IMAGES", src / "ANNOTATIONS"
    if not images_dir.is_dir() or not annots_dir.is_dir():
        raise SystemExit(f"expected IMAGES/ and ANNOTATIONS/ under {src}. "
                         f"Run scripts/fetch_dataset.py first.")

    img_by_stem = {p.stem: p for p in images_dir.glob("*.jpg")}
    ann_by_stem = {p.stem: p for p in annots_dir.glob("*.xml")}
    paired = sorted(img_by_stem.keys() & ann_by_stem.keys())
    orphans = sorted((img_by_stem.keys() ^ ann_by_stem.keys()))

    # First pass: read every annotation so the class list is discovered from the
    # data, not assumed. A silently renamed class would otherwise become class 0.
    parsed: dict[str, tuple[int, int, list]] = {}
    class_names: set[str] = set()
    for stem in paired:
        w, h, boxes = parse_voc(ann_by_stem[stem])
        parsed[stem] = (w, h, boxes)
        class_names.update(name for name, *_ in boxes)

    names = sorted(class_names)
    class_id = {name: i for i, name in enumerate(names)}

    if dst.exists():
        shutil.rmtree(dst)
    for split in SPLITS:
        (dst / split / "images").mkdir(parents=True)
        (dst / split / "labels").mkdir(parents=True)

    assignment = stratified_split(paired, ratios, seed)
    stats = {s: Counter() for s in SPLITS}
    dropped = 0
    areas: list[float] = []

    for split, stems in assignment.items():
        for stem in stems:
            w, h, boxes = parsed[stem]
            lines = []
            for name, *corners in boxes:
                yolo = to_yolo(tuple(corners), w, h)
                if yolo is None:
                    dropped += 1
                    continue
                cx, cy, bw, bh = yolo
                lines.append(f"{class_id[name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                stats[split][name] += 1
                areas.append(bw * bh)

            shutil.copy2(img_by_stem[stem], dst / split / "images" / f"{stem}.jpg")
            (dst / split / "labels" / f"{stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )

    data_yaml = {
        "path": str(dst).replace("\\", "/"),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {i: n for i, n in enumerate(names)},
    }
    (dst / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")

    return {
        "names": names,
        "assignment": {s: len(v) for s, v in assignment.items()},
        "stats": stats,
        "orphans": orphans,
        "dropped": dropped,
        "areas": sorted(areas),
        "data_yaml": dst / "data.yaml",
    }


def report(r: dict) -> None:
    names = r["names"]
    print(f"\nclasses discovered ({len(names)}): {', '.join(names)}")

    if r["orphans"]:
        print(f"\nunpaired files skipped: {len(r['orphans'])}")
    if r["dropped"]:
        print(f"degenerate boxes dropped (<1 px after clamping): {r['dropped']}")

    print(f"\n{'class':<18}" + "".join(f"{s:>8}" for s in SPLITS) + f"{'total':>8}")
    print("-" * 50)
    for name in names:
        counts = [r["stats"][s][name] for s in SPLITS]
        print(f"{name:<18}" + "".join(f"{c:>8}" for c in counts) + f"{sum(counts):>8}")
    totals = [sum(r["stats"][s].values()) for s in SPLITS]
    print("-" * 50)
    print(f"{'boxes':<18}" + "".join(f"{t:>8}" for t in totals) + f"{sum(totals):>8}")
    print(f"{'images':<18}" + "".join(f"{r['assignment'][s]:>8}" for s in SPLITS)
          + f"{sum(r['assignment'].values()):>8}")

    areas = r["areas"]
    if areas:
        median = areas[len(areas) // 2]
        small = sum(1 for a in areas if a < 0.01) / len(areas)
        print(f"\nmedian box area: {median * 100:.1f}% of the image")
        print(f"boxes under 1% of image area: {small * 100:.1f}%  "
              f"(this is what 'small object detection' means here)")

    print(f"\nwrote {r['data_yaml']}")


def demo() -> None:
    """Self-check the coordinate conversion, the part most likely to be wrong."""
    # A box filling the left half of a 200x200 image.
    assert to_yolo((0, 0, 100, 200), 200, 200) == (0.25, 0.5, 0.5, 1.0)
    # Out-of-bounds corners get clamped, never normalised past 1.0.
    cx, cy, bw, bh = to_yolo((-5, -5, 205, 205), 200, 200)
    assert (cx, cy, bw, bh) == (0.5, 0.5, 1.0, 1.0), (cx, cy, bw, bh)
    # Reversed corners are tolerated.
    assert to_yolo((100, 200, 0, 0), 200, 200) == (0.25, 0.5, 0.5, 1.0)
    # A zero-width box is dropped, not emitted as a NaN.
    assert to_yolo((50, 50, 50, 60), 200, 200) is None
    # Stratification keeps every class present in every split.
    stems = [f"{c}_{i}" for c in ("a", "b") for i in range(1, 101)]
    split = stratified_split(stems, (0.7, 0.2, 0.1), seed=0)
    for s in SPLITS:
        classes = {x.rsplit("_", 1)[0] for x in split[s]}
        assert classes == {"a", "b"}, (s, classes)
    assert sum(len(v) for v in split.values()) == len(stems)
    print("prepare_dataset self-check passed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=str(REPO_ROOT / "data" / "NEU-DET-raw"))
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "yolo"))
    ap.add_argument("--ratios", default="0.7,0.2,0.1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--self-check", action="store_true", help="run assertions and exit")
    args = ap.parse_args()

    if args.self_check:
        demo()
        return

    ratios = tuple(float(x) for x in args.ratios.split(","))
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        raise SystemExit(f"--ratios must be three numbers summing to 1, got {args.ratios}")

    report(convert(Path(args.src), Path(args.out), ratios, args.seed))


if __name__ == "__main__":
    main()
