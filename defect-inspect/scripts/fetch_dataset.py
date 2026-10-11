"""Fetch the NEU-DET dataset (1800 VOC annotations, 200x200 grayscale steel images).

Why a sparse git clone and not a plain one? The upstream repo is 423 MB of
notebooks, caches and checkpoints. We want two folders totalling ~45 MB.
A partial clone (--filter=blob:none) downloads no file contents up front, then
sparse-checkout pulls blobs for only the paths we ask for.

Run:  uv run scripts/fetch_dataset.py
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "https://github.com/siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection.git"
WANTED = ["IMAGES", "ANNOTATIONS"]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd, text=True)
    if result.returncode != 0:
        raise SystemExit(f"command failed with exit code {result.returncode}")


def fetch(dest: Path, force: bool) -> None:
    if dest.exists():
        if not force:
            print(f"{dest} already exists. Use --force to re-download.")
            return
        shutil.rmtree(dest)

    dest.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--filter=blob:none", "--sparse", "--depth", "1",
         UPSTREAM, str(dest)])
    run(["git", "sparse-checkout", "set", *WANTED], cwd=dest)


def verify(dest: Path) -> None:
    images = sorted((dest / "IMAGES").glob("*.jpg"))
    annots = sorted((dest / "ANNOTATIONS").glob("*.xml"))
    print(f"\nimages     : {len(images)}")
    print(f"annotations: {len(annots)}")

    if not images or not annots:
        raise SystemExit("fetch produced an empty dataset; check the network and retry")

    img_stems = {p.stem for p in images}
    ann_stems = {p.stem for p in annots}
    orphan_ann = sorted(ann_stems - img_stems)
    orphan_img = sorted(img_stems - ann_stems)

    # Upstream ships 1800 XML but only 1770 JPG. Report it loudly instead of
    # letting the training loader silently drop 30 samples.
    if orphan_ann:
        print(f"\n{len(orphan_ann)} annotations have no image (will be skipped):")
        print("  " + ", ".join(orphan_ann[:10]) + (" ..." if len(orphan_ann) > 10 else ""))
    if orphan_img:
        print(f"\n{len(orphan_img)} images have no annotation (will be skipped):")
        print("  " + ", ".join(orphan_img[:10]) + (" ..." if len(orphan_img) > 10 else ""))
    if not orphan_ann and not orphan_img:
        print("\nevery image has an annotation")

    print(f"\nusable pairs: {len(img_stems & ann_stems)}")
    print(f"raw dataset ready at: {dest}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", default=str(REPO_ROOT / "data" / "NEU-DET-raw"))
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    dest = Path(args.dest)
    fetch(dest, args.force)
    verify(dest)


if __name__ == "__main__":
    sys.exit(main())
