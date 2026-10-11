"""Build the PatchCore memory bank, and export its backbone to ONNX.

PatchCore in one paragraph: push images through a pretrained CNN, keep the
intermediate feature vectors ("patch embeddings") for every spatial location,
and store a representative subset of the ones that came from normal surfaces.
At inference, embed the new image the same way and measure how far each patch
is from its nearest stored neighbour. Far from everything normal means anomalous.
Nothing is trained. See docs/14_patchcore_anomaly.md.

--------------------------------------------------------------------------
THE PROBLEM WITH THIS DATASET, STATED PLAINLY

PatchCore is defined to train on defect-free images. NEU-DET contains none --
all 1770 images have at least one labelled defect. There is no "good parts"
folder to point at.

The workaround here: build the bank from the regions of each image that lie
OUTSIDE every annotated bounding box. Those regions are, by the annotation's
own claim, normal steel surface. It gives roughly 100k normal patches from
images that are nominally all defective.

The honest limitations of that, which belong in the viva answer:

  1. Bounding boxes are rectangles around irregular defects, so the corners of
     a box contain normal surface and are excluded anyway -- that direction is
     safe. The unsafe direction is defects that extend past their box, which
     leaks defect texture into the bank and makes the model tolerant of it.
  2. Crazing covers much of its image, so few of its patches survive the mask.
     The bank under-represents the crazing background texture.
  3. A real deployment would photograph 200 known-good parts under the
     production lighting and skip all of this. That is the correct fix, and it
     needs the conveyor rig.

Run:  uv run scripts/build_patchcore.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

# ImageNet statistics. The backbone was pretrained with these, so inference must
# reproduce them exactly or every feature vector is shifted.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_backbone(imgsz: int, out_path: Path) -> tuple[int, int]:
    """Export ResNet-18 layers 2 and 3, fused, to ONNX.

    Why ResNet-18 and not the WideResNet-50 the paper uses: the paper optimised
    for accuracy on a workstation, we have four ARM cores sharing a frame budget
    with YOLOv8. ResNet-18 is roughly 10x cheaper and its mid-level features are
    still generic texture descriptors, which is all PatchCore needs.

    Why layers 2 and 3 and not the last layer: deep features encode object
    identity ("this is a cat"), which is meaningless for a texture. Mid-level
    features encode local appearance, which is exactly the signal here. Layer 2
    keeps spatial detail, layer 3 sees more context; concatenating gives both.

    Exporting to ONNX is what keeps torch off the Pi. The bank is built once,
    here, on the laptop; the Pi only ever runs the ONNX graph.
    """
    import torch
    import torch.nn as nn
    from torchvision.models import ResNet18_Weights, resnet18

    class PatchBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
            self.layer1, self.layer2, self.layer3 = net.layer1, net.layer2, net.layer3

        def forward(self, x):
            x = self.layer1(self.stem(x))
            f2 = self.layer2(x)                     # stride 8,  128 channels
            f3 = self.layer3(f2)                    # stride 16, 256 channels
            f3 = nn.functional.interpolate(f3, size=f2.shape[-2:], mode="bilinear",
                                           align_corners=False)
            f = torch.cat([f2, f3], dim=1)          # 384 channels at stride 8
            # Average pooling over 3x3 with padding is what the paper calls
            # local aggregation: each patch descriptor absorbs its neighbours,
            # which makes the descriptor tolerant of a one-pixel shift.
            f = nn.functional.avg_pool2d(f, 3, stride=2, padding=1)
            return f

    model = PatchBackbone().eval()
    dummy = torch.zeros(1, 3, imgsz, imgsz)
    with torch.no_grad():
        out = model(dummy)
    _, channels, grid, _ = out.shape

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, dummy, str(out_path), opset_version=13,
                      input_names=["images"], output_names=["features"],
                      dynamo=False)
    print(f"backbone -> {out_path}  ({channels} channels, {grid}x{grid} grid, "
          f"{grid * grid} patches per image)")
    return channels, grid


def preprocess_for_backbone(img: np.ndarray, imgsz: int) -> np.ndarray:
    """ImageNet preprocessing. Different from the detector's, deliberately."""
    resized = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
    x = resized[:, :, ::-1].astype(np.float32) / 255.0        # BGR -> RGB, 0..1
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.ascontiguousarray(x.transpose(2, 0, 1))
    return x[np.newaxis, ...]


def normal_mask(label_path: Path, grid: int, dilate: float) -> np.ndarray:
    """True where a grid cell contains no annotated defect.

    The YOLO label is normalised, so it maps onto the feature grid without
    knowing the image size. Boxes are dilated before masking because a defect
    bleeds past its annotation and the feature at a cell touching a defect is
    contaminated by it.
    """
    mask = np.ones((grid, grid), dtype=bool)
    if not label_path.exists():
        return mask

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, w, h = (float(p) for p in parts)
        w, h = w * dilate, h * dilate
        x1 = int(np.floor((cx - w / 2) * grid))
        x2 = int(np.ceil((cx + w / 2) * grid))
        y1 = int(np.floor((cy - h / 2) * grid))
        y2 = int(np.ceil((cy + h / 2) * grid))
        mask[max(0, y1):min(grid, y2), max(0, x1):min(grid, x2)] = False
    return mask


def extract(session, images: list[Path], labels_dir: Path, imgsz: int, grid: int,
            dilate: float) -> tuple[np.ndarray, int, int]:
    """Collect defect-free patch embeddings across the training images."""
    input_name = session.get_inputs()[0].name
    kept, total = [], 0

    for i, path in enumerate(images):
        img = cv2.imread(str(path))
        if img is None:
            continue
        feats = session.run(None, {input_name: preprocess_for_backbone(img, imgsz)})[0]
        # (1, C, G, G) -> (G*G, C): one row per spatial location.
        patches = feats[0].reshape(feats.shape[1], -1).T
        mask = normal_mask(labels_dir / f"{path.stem}.txt", grid, dilate).reshape(-1)
        total += patches.shape[0]
        if mask.any():
            kept.append(patches[mask])
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(images)} images, "
                  f"{sum(len(k) for k in kept):,} normal patches so far")

    stacked = np.concatenate(kept, axis=0).astype(np.float32)
    return stacked, stacked.shape[0], total


def denoise(features: np.ndarray, frac: float, k: int = 8,
            probe: int = 4000, seed: int = 42) -> np.ndarray:
    """Drop the `frac` most isolated patches before coreset selection.

    Why this is needed, and why it fights the next function.

    The bank is built from regions outside every annotated box, on the
    annotation's claim that they are normal steel. NEU-DET's boxes are loose and
    its subtle defects are documented as incompletely annotated, so some defect
    texture survives the mask and enters the "normal" pool. A memory bank that
    contains defect patches scores defects as normal -- the failure mode SoftPatch
    (Jiang et al., NeurIPS 2022) names: "the naively attained memory bank contains
    abnormal patch features, and assigns a low anomaly score on defect images".

    That leak is made worse, not better, by `coreset` below: greedy k-centre
    deliberately selects the extremes of the pool, and a leaked defect patch is
    exactly an extreme. The two steps pull in opposite directions, so the filter
    has to run first.

    Isolation is measured as the distance to the k-th nearest neighbour within a
    random probe subset -- genuine mill finish is dense and has close neighbours,
    a leaked defect patch does not. The probe keeps this O(N * probe) instead of
    O(N^2) on a pool of 100k.

    frac=0 returns the input untouched, which is the pre-existing behaviour.
    """
    if frac <= 0:
        return features
    rng = np.random.default_rng(seed)
    n = features.shape[0]
    idx = rng.choice(n, size=min(probe, n), replace=False)
    ref = features[idx]

    kth = np.empty(n, dtype=np.float32)
    step = 2048
    for i in range(0, n, step):
        chunk = features[i:i + step]
        d = np.linalg.norm(chunk[:, None, :] - ref[None, :, :], axis=2)
        # k+1 because a patch that is in the probe set matches itself at 0.
        kth[i:i + step] = np.partition(d, k, axis=1)[:, k]

    keep = kth <= np.quantile(kth, 1.0 - frac)
    print(f"  denoise: dropped {(~keep).sum():,} of {n:,} patches "
          f"({frac:.0%} most isolated, kth-NN cutoff {np.quantile(kth, 1.0 - frac):.3f})")
    return features[keep]


def coreset(features: np.ndarray, size: int, seed: int) -> np.ndarray:
    """Greedy k-centre subsample: pick points that are far from everything picked.

    Why not just take a random sample: the patch set is dominated by the most
    common texture, so a random sample is almost entirely mill finish and the
    rare-but-normal appearances (an edge, a lighting gradient) get no
    representation. Those are exactly the patches that would otherwise be
    flagged as anomalies. Greedy k-centre deliberately picks the outliers of the
    normal set, which is what makes a 2000-vector bank behave like a 100k one.

    Each step is a distance computation against all candidates, so this is
    O(size x N). That is why the candidate pool is capped first.
    """
    rng = np.random.default_rng(seed)
    n = features.shape[0]
    size = min(size, n)

    selected = np.empty(size, dtype=np.int64)
    selected[0] = rng.integers(n)
    # Distance from every candidate to the nearest already-selected point.
    min_dist = np.linalg.norm(features - features[selected[0]], axis=1)

    for i in range(1, size):
        # The farthest point from the current selection is the most
        # under-represented, so it becomes the next centre.
        pick = int(np.argmax(min_dist))
        selected[i] = pick
        np.minimum(min_dist, np.linalg.norm(features - features[pick], axis=1),
                   out=min_dist)
        if (i + 1) % 500 == 0:
            print(f"  coreset {i + 1}/{size}, current max distance {min_dist.max():.3f}")

    return features[selected]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", default=str(REPO_ROOT / "data" / "yolo" / "train" / "images"))
    ap.add_argument("--labels", default=str(REPO_ROOT / "data" / "yolo" / "train" / "labels"))
    ap.add_argument("--imgsz", type=int, default=256)
    ap.add_argument("--bank-size", type=int, default=2000,
                    help="vectors kept after coreset subsampling")
    ap.add_argument("--pool", type=int, default=40000,
                    help="candidate patches fed to the coreset selector")
    ap.add_argument("--proj-dim", type=int, default=128,
                    help="random-projection target dimension, 0 to disable")
    ap.add_argument("--dilate", type=float, default=1.3,
                    help="box expansion before masking, so defect bleed is excluded")
    ap.add_argument("--max-images", type=int, default=600)
    ap.add_argument("--denoise", type=float, default=0.0,
                    help="fraction of most-isolated patches to drop before "
                         "coreset; 0 reproduces the original bank")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    images_dir, labels_dir = Path(args.images), Path(args.labels)
    images = sorted(images_dir.glob("*.jpg"))
    if not images:
        raise SystemExit(f"no images in {images_dir}; run scripts/prepare_dataset.py")

    rng = np.random.default_rng(args.seed)
    if len(images) > args.max_images:
        idx = rng.choice(len(images), args.max_images, replace=False)
        images = [images[i] for i in sorted(idx)]

    backbone_path = REPO_ROOT / "models" / "patchcore_backbone.onnx"
    channels, grid = build_backbone(args.imgsz, backbone_path)

    import onnxruntime as ort

    session = ort.InferenceSession(str(backbone_path), providers=["CPUExecutionProvider"])

    print(f"\nextracting defect-free patches from {len(images)} images "
          f"(boxes dilated {args.dilate}x before masking)")
    features, kept, total = extract(session, images, labels_dir, args.imgsz,
                                    grid, args.dilate)
    print(f"kept {kept:,} of {total:,} patches ({kept / total * 100:.1f}% outside "
          f"every annotated box)")

    # Random projection before the coreset. Johnson-Lindenstrauss says a random
    # linear map into a lower dimension preserves pairwise distances to within a
    # small factor, and distance is the only thing PatchCore uses. This makes
    # both the coreset search and every runtime lookup 3x cheaper.
    projection = None
    if args.proj_dim and args.proj_dim < features.shape[1]:
        projection = rng.normal(
            0, 1 / np.sqrt(args.proj_dim), (features.shape[1], args.proj_dim)
        ).astype(np.float32)
        features = features @ projection
        print(f"projected {channels} -> {args.proj_dim} dimensions")

    # Filter before sampling and before coreset: k-centre selects extremes, and
    # a leaked defect patch is an extreme, so the order matters.
    if args.denoise > 0:
        print(f"\ndenoising the normal pool")
        features = denoise(features, args.denoise, seed=args.seed)

    if features.shape[0] > args.pool:
        idx = rng.choice(features.shape[0], args.pool, replace=False)
        features = features[idx]
        print(f"sampled {args.pool:,} candidates for coreset selection")

    print(f"\nselecting a {args.bank_size}-vector coreset")
    bank = coreset(features, args.bank_size, args.seed)

    out = REPO_ROOT / "models" / "patchcore_bank.npz"
    np.savez_compressed(
        out, bank=bank,
        projection=projection if projection is not None else np.zeros(0, np.float32),
        imgsz=np.int32(args.imgsz), grid=np.int32(grid), channels=np.int32(channels),
    )
    print(f"\nbank -> {out}")
    print(f"  {bank.shape[0]} vectors x {bank.shape[1]} dims, "
          f"{out.stat().st_size / 1e6:.2f} MB")
    print("\nnext: uv run python -m inspector.anomaly   (calibrates the threshold)")


if __name__ == "__main__":
    main()
