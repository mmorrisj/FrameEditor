"""Image fingerprints and feature vectors.

- phash: 64-bit DCT perceptual hash. Survives re-encoding, rescaling and mild
  color shifts; compared by Hamming distance. Used for duplicate detection.
- embed(images, backend): L2-normalized vectors compared by cosine similarity.
    "visual": built-in, CPU-only, no model download. A 16x16 grayscale layout
              plus an HSV color histogram. Groups by look and composition.
    "clip":   CLIP image embeddings via transformers + torch (optional
              install, GPU used when available). Groups by content.
"""
from __future__ import annotations

import importlib.util
import os
import threading

import numpy as np
from PIL import Image
from scipy.fft import dctn

_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
LOW_DETAIL_STD = 4.0  # grayscale std below this = blank/flat frame, hash unreliable


# --- perceptual hash ----------------------------------------------------------

def phash_gray(g: np.ndarray) -> np.ndarray:
    """(32,32) or (n,32,32) grayscale -> uint64 hash(es)."""
    g = np.asarray(g, dtype=np.float32)
    single = g.ndim == 2
    if single:
        g = g[None]
    d = dctn(g, axes=(-2, -1), norm="ortho")[:, :8, :8].reshape(len(g), 64)
    med = np.median(d[:, 1:], axis=1, keepdims=True)  # DC term would skew the median
    bits = np.packbits(d > med, axis=1)                # (n, 8) uint8
    h = bits.view(">u8").astype(np.uint64).reshape(-1)
    return h[0] if single else h


def gray32(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L").resize((32, 32), Image.BOX), dtype=np.float32)


def phash_image(img: Image.Image) -> tuple[int, bool]:
    """(hash, reliable) for one image; unreliable = nearly uniform frame."""
    g = gray32(img)
    return int(phash_gray(g)), bool(g.std() >= LOW_DETAIL_STD)


def detail_mask(g: np.ndarray) -> np.ndarray:
    """(n,32,32) grayscale -> bool (n,) True where the frame has real detail."""
    return np.asarray(g, dtype=np.float32).reshape(len(g), -1).std(axis=1) >= LOW_DETAIL_STD


def hamming(a, b) -> np.ndarray:
    """Bit distance between uint64 hash arrays (broadcasting)."""
    x = np.bitwise_xor(np.asarray(a, dtype=np.uint64), np.asarray(b, dtype=np.uint64))
    shape = x.shape
    x = np.ascontiguousarray(x.reshape(-1))
    return _POP[x.view(np.uint8)].reshape(shape + (8,)).sum(axis=-1, dtype=np.int64)


# --- embeddings ---------------------------------------------------------------

def visual_vector(img: Image.Image) -> np.ndarray:
    small = img.convert("RGB").resize((64, 64), Image.BOX)
    g = np.asarray(small.convert("L").resize((16, 16), Image.BOX), dtype=np.float32).reshape(-1)
    g -= g.mean()
    g /= (np.linalg.norm(g) or 1.0)
    hsv = np.asarray(small.convert("HSV"), dtype=np.int32).reshape(-1, 3)
    idx = (hsv[:, 0] * 8 // 256) * 16 + (hsv[:, 1] * 4 // 256) * 4 + (hsv[:, 2] * 4 // 256)
    hist = np.sqrt(np.bincount(idx, minlength=128).astype(np.float32))
    hist /= (np.linalg.norm(hist) or 1.0)
    v = np.concatenate([g, hist])
    return (v / (np.linalg.norm(v) or 1.0)).astype(np.float32)


class _Clip:
    _inst = None
    _lock = threading.Lock()

    def __init__(self):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        name = os.environ.get("FRAMEKIT_CLIP_MODEL", "openai/clip-vit-base-patch32")
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(name).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(name)

    @classmethod
    def get(cls) -> "_Clip":
        with cls._lock:
            if cls._inst is None:
                cls._inst = cls()
            return cls._inst

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for i in range(0, len(images), 32):
                batch = self.proc(images=images[i:i + 32], return_tensors="pt").to(self.device)
                f = self.model.get_image_features(**batch)
                if not isinstance(f, self.torch.Tensor):  # transformers 5 returns an output object
                    f = f.pooler_output                   # the projected image embedding
                f = f / f.norm(dim=-1, keepdim=True)
                out.append(f.float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 512), np.float32)


def backends() -> dict[str, bool]:
    clip_ok = all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers"))
    return {"visual": True, "clip": clip_ok}


def embed(images: list[Image.Image], backend: str = "visual") -> np.ndarray:
    if backend == "visual":
        return np.stack([visual_vector(i) for i in images]) if images else np.zeros((0, 384), np.float32)
    if backend == "clip":
        if not backends()["clip"]:
            raise RuntimeError("the clip backend needs torch and transformers installed")
        return _Clip.get().embed(images)
    raise ValueError(f"unknown backend {backend!r}")
