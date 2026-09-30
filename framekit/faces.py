"""Face identity: find faces and compare them, to catch AI identity drift.

Uses InsightFace's buffalo_l models (SCRFD detector + ArcFace recogniser),
run locally with onnxruntime, on the GPU when one is available. The models
are downloaded once to ~/.insightface/models and nothing else leaves the
machine. They are licensed for non-commercial use.

Likeness is the cosine similarity of two ArcFace embeddings: the same face
typically scores 0.6-0.9 across poses and lighting, different people rarely
above 0.3. The bands below turn that into on model / drifting / off model.

Install (optional; everything else in FrameKit works without it):
    pip install --no-deps insightface==2.0
    pip install -r requirements-faces.txt
"""
from __future__ import annotations

import importlib.util
import os
import threading

import numpy as np

MODEL = os.environ.get("FRAMEKIT_FACE_MODEL", "buffalo_l")
ON_MODEL = 0.55          # at or above: the same character
DRIFTING = 0.40          # at or above (below ON_MODEL): drifting; below: off model
MAX_FACES = 4            # faces kept per frame (largest first)


def available() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("insightface", "onnxruntime"))


def band(score: float | None) -> str | None:
    if score is None:
        return None
    return "on" if score >= ON_MODEL else "drift" if score >= DRIFTING else "off"


class _App:
    _inst = None
    _lock = threading.Lock()

    def __init__(self):
        try:  # torch's CUDA libraries let onnxruntime use the GPU without a separate CUDA install
            import torch  # noqa: F401
            gpu = torch.cuda.is_available()
        except ImportError:
            gpu = False
        from insightface.app import FaceAnalysis
        providers = (["CUDAExecutionProvider"] if gpu else []) + ["CPUExecutionProvider"]
        self.app = FaceAnalysis(name=MODEL, allowed_modules=["detection", "recognition"], providers=providers)
        self.app.prepare(ctx_id=0 if gpu else -1, det_size=(640, 640))
        self.run_lock = threading.Lock()  # one inference at a time on a shared session

    @classmethod
    def get(cls) -> "_App":
        if not available():
            raise RuntimeError("face checks need insightface and onnxruntime: "
                               "pip install --no-deps insightface==2.0 && pip install -r requirements-faces.txt")
        with cls._lock:
            if cls._inst is None:
                cls._inst = cls()
            return cls._inst


def detect(img: np.ndarray) -> list[dict]:
    """Faces in an RGB uint8 image, largest first: [{box, score, emb (512, unit length)}]."""
    app = _App.get()
    with app.run_lock:
        found = app.app.get(np.ascontiguousarray(img[:, :, ::-1]))  # insightface expects BGR
    out = []
    for f in found:
        x1, y1, x2, y2 = (float(v) for v in f.bbox)
        out.append({"box": [round(x1), round(y1), round(x2), round(y2)], "score": round(float(f.det_score), 3),
                    "area": (x2 - x1) * (y2 - y1), "emb": f.normed_embedding.astype(np.float32)})
    out.sort(key=lambda f: -f["area"])
    return out[:MAX_FACES]


def likeness(ref: np.ndarray, faces: list[np.ndarray]) -> float | None:
    """Best match of `ref` among a frame's faces (the character may share the frame)."""
    if ref is None or not len(faces):
        return None
    return float(np.max(np.stack(faces) @ ref))
