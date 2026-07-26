"""Region edits on extracted frames (Pillow).

Before a frame's first edit, its pristine copy is saved to pristine/ — that
both powers revert and marks the frame as "edited" in the UI. Edits after the
first stack on the working frame; revert always returns to the original.

Rectangles arrive normalized (0..1 relative to frame size) so the frontend
never needs to know the video's real resolution.
"""
from __future__ import annotations

import os
import shutil

from PIL import Image, ImageFilter

OPS = ("blur", "pixelate", "clone")


def _frame_path(d: str, n: int) -> str:
    return os.path.join(d, "frames", f"f{n:06d}.png")


def _pristine_path(d: str, n: int) -> str:
    return os.path.join(d, "pristine", f"f{n:06d}.png")


def _backup(d: str, n: int) -> None:
    dst = _pristine_path(d, n)
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(_frame_path(d, n), dst)


def _box(rect: dict, w: int, h: int) -> tuple[int, int, int, int]:
    """Normalized rect -> clamped pixel box, at least 1x1."""
    x0 = max(0, min(w - 1, round(rect["x"] * w)))
    y0 = max(0, min(h - 1, round(rect["y"] * h)))
    x1 = max(x0 + 1, min(w, round((rect["x"] + rect["w"]) * w)))
    y1 = max(y0 + 1, min(h, round((rect["y"] + rect["h"]) * h)))
    return x0, y0, x1, y1


def apply(d: str, n: int, rect: dict, op: str, strength: int, total_frames: int) -> None:
    if op not in OPS:
        raise ValueError(f"unknown op {op!r}")
    path = _frame_path(d, n)
    if not os.path.exists(path):
        raise FileNotFoundError(f"frame {n} does not exist")
    _backup(d, n)

    img = Image.open(path).convert("RGB")
    box = _box(rect, *img.size)
    x0, y0, x1, y1 = box
    rw, rh = x1 - x0, y1 - y0

    if op == "clone":
        # Copy the region from the neighboring frame's *current* state. Applied
        # ascending over a range, each bad frame inherits the already-cleaned
        # pixels of the one before it, so a fix cascades through the range.
        src_n = n - 1 if n > 1 else n + 1
        if not 1 <= src_n <= total_frames:
            raise ValueError("no neighboring frame to clone from")
        with Image.open(_frame_path(d, src_n)).convert("RGB") as src:
            region = src.crop(box)
    else:
        region = img.crop(box)
        if op == "blur":
            region = region.filter(ImageFilter.GaussianBlur(radius=max(2, strength)))
        else:  # pixelate
            f = max(2, strength)
            region = region.resize(
                (max(1, rw // f), max(1, rh // f)), Image.BILINEAR
            ).resize((rw, rh), Image.NEAREST)

    img.paste(region, (x0, y0))
    img.save(path)


def revert(d: str, n: int) -> None:
    src = _pristine_path(d, n)
    if not os.path.exists(src):
        return  # never edited; nothing to do
    shutil.copy2(src, _frame_path(d, n))
    os.remove(src)


def replace(d: str, n: int, file_storage, size: tuple[int, int]) -> None:
    """Swap in an externally edited frame (the GIMP escape hatch).

    Whatever arrives is normalized to the video's exact resolution and RGB PNG,
    so the repackage step can trust every frame in the sequence.
    """
    _backup(d, n)
    img = Image.open(file_storage.stream).convert("RGB")
    if img.size != size:
        img = img.resize(size, Image.LANCZOS)
    img.save(_frame_path(d, n))
