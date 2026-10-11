"""Camera frame -> model input tensor, and detection boxes -> original pixels.

Four conversions separate a camera frame from what the network expects, and
every one of them fails silently if you get it wrong: the model still runs, it
still emits numbers, and it detects nothing. See docs/03_numpy_images.md.

  (H, W, 3) uint8 BGR   ->   (1, 3, S, S) float32 RGB in 0..1

The letterbox offsets are returned because a box found in the padded 320x320
image means nothing until it is mapped back to a pixel in the original frame.
"""

from __future__ import annotations

import cv2
import numpy as np

# The grey ultralytics pads with during training. Matching it means the model
# sees the same padding distribution at inference that it learned on.
PAD_VALUE = 114


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize to a square, preserving aspect ratio, padding the remainder.

    Returns (padded_image, scale, (pad_left, pad_top)).
    """
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        raise ValueError(f"empty frame with shape {img.shape}")

    scale = min(size / h, size / w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))

    # INTER_AREA is the correct filter when shrinking: it averages the pixels
    # being merged instead of point-sampling, which is what stops a thin
    # scratch from vanishing between sample points.
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)

    canvas = np.full((size, size, 3), PAD_VALUE, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, scale, (left, top)


def to_tensor(img: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 BGR -> (1, 3, H, W) float32 RGB in 0..1."""
    x = img[:, :, ::-1]                  # BGR -> RGB; a view with negative stride
    x = x.astype(np.float32) / 255.0     # this materialises a real array
    x = x.transpose(2, 0, 1)             # HWC -> CHW; a view again, non-contiguous
    x = np.ascontiguousarray(x)          # ONNX Runtime needs contiguous memory
    return x[np.newaxis, ...]            # -> NCHW


def preprocess(frame: np.ndarray, size: int):
    """Full camera-to-model path. Returns (tensor, scale, pad)."""
    padded, scale, pad = letterbox(frame, size)
    return to_tensor(padded), scale, pad


def unletterbox_boxes(boxes: np.ndarray, scale: float, pad: tuple[int, int],
                      orig_shape: tuple[int, int]) -> np.ndarray:
    """Map xyxy boxes from letterboxed coordinates back to the original frame.

    Inverse of letterbox(): subtract the padding, then undo the scale. Clipping
    at the end matters because a box may legitimately extend past the edge of
    the visible part after the model's own regression.
    """
    if boxes.size == 0:
        return boxes

    left, top = pad
    h, w = orig_shape[:2]
    out = boxes.astype(np.float32).copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - left) / scale
    out[:, [1, 3]] = (out[:, [1, 3]] - top) / scale
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, w)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, h)
    return out


def demo() -> None:
    """Self-check: the round trip must land back where it started.

    This is the test that catches the silent preprocessing bug. If letterbox and
    unletterbox_boxes ever disagree, boxes drift and nothing else reports it.
    """
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    padded, scale, pad = letterbox(frame, 320)

    assert padded.shape == (320, 320, 3), padded.shape
    assert abs(scale - 320 / 640) < 1e-6, scale
    # 640 wide scales to 320, 480 tall scales to 240, so 40 px of pad top and bottom.
    assert pad == (0, 40), pad
    assert padded[0, 0].tolist() == [PAD_VALUE] * 3, "top band should be padding"

    tensor, _, _ = preprocess(frame, 320)
    assert tensor.shape == (1, 3, 320, 320), tensor.shape
    assert tensor.dtype == np.float32, tensor.dtype
    assert 0.0 <= tensor.min() and tensor.max() <= 1.0
    assert tensor.flags["C_CONTIGUOUS"], "ONNX Runtime requires contiguous input"

    # Channel order: a pure-red BGR pixel must land in channel 0 (R) of the tensor.
    red = np.zeros((10, 10, 3), dtype=np.uint8)
    red[:, :, 2] = 255                      # BGR: index 2 is red
    t = to_tensor(red)
    assert t[0, 0].max() == 1.0, "red should be channel 0 after BGR->RGB"
    assert t[0, 2].max() == 0.0, "blue channel should be empty"

    # Round trip: a known box in original coordinates -> letterboxed -> back.
    orig = np.array([[100.0, 200.0, 300.0, 400.0]])
    lb = orig.copy()
    lb[:, [0, 2]] = lb[:, [0, 2]] * scale + pad[0]
    lb[:, [1, 3]] = lb[:, [1, 3]] * scale + pad[1]
    back = unletterbox_boxes(lb, scale, pad, frame.shape)
    assert np.allclose(back, orig, atol=1e-3), (back, orig)

    # A non-square shrink in the other direction.
    tall = np.zeros((640, 480, 3), dtype=np.uint8)
    _, s2, p2 = letterbox(tall, 320)
    assert p2 == (40, 0), p2

    print("preprocess self-check passed")


if __name__ == "__main__":
    demo()
