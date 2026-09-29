"""Color match: undo the color drift that builds up across chained AI video segments.

Each generation hop (VAE encode, generate, VAE decode) nudges color and
contrast, often darker and more saturated, and the next segment inherits the
shifted handoff frame, so the error compounds. This tool corrects each
segment and joins them:

  1. Load the segments in order. Each one usually starts with the previous
     segment's last frame (the handoff); that overlap is detected per join.
  2. Fit ONE color transform per segment and apply it to every frame of that
     segment. Per-frame matching flickers and erases lighting changes you
     want (walking into shade); a single transform keeps them.
       seam mode:     (default) fit the segment's overlap frames to the
                      corrected copy of the same frames at the end of the
                      previous segment. Same moment, same content, so any
                      difference is pure drift; lighting that changes across
                      segments survives. Segment 1 is fitted to the original.
       original mode: fit every segment's first few frames to the original
                      start image. Only sensible when the shot's colors stay
                      put; if the content itself changes color (new scenery,
                      a sunset), this drags it back toward the start image.
                      In tests with moving content it made things worse,
                      while seam mode cut the error by half or more.
  3. Drop the repeated overlap frames at each join (keeping the earlier
     segment's copy, one fewer VAE pass), optionally crossfade, and encode
     once. Frames travel as PNG between decode and encode, and the YUV<->RGB
     conversion is pinned to the source's matrix and range on the way in and
     BT.709 limited range on the way out, so the pipeline adds no shift of
     its own.

Methods (all fitted once, then applied as a fixed mapping):
  exact       (default) the handoff frame appears in both clips, so its pixels
              correspond one to one; fit a cubic colour mapping from them, so
              each tone and hue can shift its own way. On chains with known
              ground truth it beat HM-MVGD-HM by 25-30%. Where there's no
              matching frame (no overlap), that segment uses HM-MVGD-HM.
  mkl         Monge-Kantorovich linear transfer: a 3x3 matrix + offset that
              maps the color distribution's mean and covariance.
  mvgd        multivariate Gaussian transfer; also a 3x3 matrix + offset.
  hm          per-channel histogram matching (lookup tables).
  hm-mvgd-hm  / hm-mkl-hm: histogram match, Gaussian transfer, histogram
              match again (the compound methods from color-matcher/KJNodes).
              The best of the distribution methods in tests.

Brightness and colour strengths: the correction is split in Oklab (a
perceptual colour space) into its lightness and colour parts, which can be
applied by different amounts, e.g. fix a colour cast but keep the brightness.

Frame rate: by default the output uses the rate most segments already have,
so the fewest clips are converted (a 16 fps clip converted to 24 fps repeats
frames and stutters).
"""
from __future__ import annotations

import io
import json
import os
import re
import secrets
import shutil
import subprocess
import time

import numpy as np
from PIL import Image

from .. import config, media

METHODS = {"exact": "Exact seam fit", "hm-mvgd-hm": "HM-MVGD-HM", "mkl": "MKL", "hm-mkl-hm": "HM-MKL-HM",
           "mvgd": "MVGD", "hm": "Histogram matching"}
MODES = {"seam": "Match each segment to the end of the previous one (recommended)",
         "original": "Match every segment to the original image (only for shots whose colors stay put)"}
SMALL_W = 256          # analysis resolution
MAX_OVERLAP = 8        # longest overlap looked for at a join
OVERLAP_MAX_ERR = 0.35 # z-scored gray difference; above this, a join has no overlap
_NOWIN = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
SID_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{4}$")


def _root(*parts: str) -> str:
    return os.path.join(config.WORK, "colormatch", *parts)


# ==== transforms ============================================================================

def _sqrtm(m: np.ndarray, inv: bool = False) -> np.ndarray:
    w, v = np.linalg.eigh((m + m.T) / 2)
    w = np.clip(w, 1e-8, None)
    return (v * (w ** (-0.5 if inv else 0.5))) @ v.T


def _stats(x: np.ndarray):
    x = x.astype(np.float64)  # float32 sums over ~10^5 pixels lose precision
    return x.mean(axis=0), np.cov(x, rowvar=False) + np.eye(3) * 1e-7


def _fit_mkl(s: np.ndarray, r: np.ndarray) -> dict:
    ms, cs = _stats(s)
    mr, cr = _stats(r)
    cs_h, cs_ih = _sqrtm(cs), _sqrtm(cs, inv=True)
    t = cs_ih @ _sqrtm(cs_h @ cr @ cs_h) @ cs_ih  # symmetric
    return {"kind": "affine", "A": t.tolist(), "b": (mr - ms @ t).tolist()}


def _fit_mvgd(s: np.ndarray, r: np.ndarray) -> dict:
    ms, cs = _stats(s)
    mr, cr = _stats(r)
    t = (_sqrtm(cr) @ _sqrtm(cs, inv=True)).T  # row-vector convention: x @ t
    return {"kind": "affine", "A": t.tolist(), "b": (mr - ms @ t).tolist()}


def _fit_hm(s: np.ndarray, r: np.ndarray, levels: int = 257) -> dict:
    q = np.linspace(0, 1, levels)
    xs, ys = [], []
    for c in range(3):
        a, b = np.quantile(s[:, c], q), np.quantile(r[:, c], q)
        a, keep = np.unique(a, return_index=True)  # np.interp needs increasing x
        xs.append(a.tolist())
        ys.append(b[keep].tolist())
    return {"kind": "lut", "x": xs, "y": ys}


def _apply_stage(st: dict, x: np.ndarray) -> np.ndarray:
    if st["kind"] == "affine":
        return x @ np.asarray(st["A"], np.float32) + np.asarray(st["b"], np.float32)
    if st["kind"] == "poly":
        return _poly(x, st.get("degree", 2)) @ np.asarray(st["beta"], np.float32)
    out = np.empty_like(x)
    for c in range(3):
        out[:, c] = np.interp(x[:, c], st["x"][c], st["y"][c])
    return out


def apply_pixels(stages: list[dict], x: np.ndarray) -> np.ndarray:
    for st in stages:
        x = _apply_stage(st, x)
    return x


def fit(src: np.ndarray, ref: np.ndarray, method: str = "mkl") -> list[dict]:
    """Fit a fixed color mapping from src pixels to ref pixels, both (N, 3) floats in 0..1."""
    if method not in METHODS or method == "exact":
        raise ValueError(f"method must be one of {[m for m in METHODS if m != 'exact']}")
    stages = []
    plan = {"mkl": ["mkl"], "mvgd": ["mvgd"], "hm": ["hm"],
            "hm-mvgd-hm": ["hm", "mvgd", "hm"], "hm-mkl-hm": ["hm", "mkl", "hm"]}[method]
    cur = src
    for step in plan:
        st = {"mkl": _fit_mkl, "mvgd": _fit_mvgd, "hm": _fit_hm}[step](cur, ref)
        stages.append(st)
        cur = np.clip(_apply_stage(st, cur), 0, 1)
    return stages


# ---- exact fit: the handoff frame appears in both clips, so pixels correspond one to one ----

def _poly(x: np.ndarray, degree: int = 2) -> np.ndarray:
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    cols = [np.ones_like(r), r, g, b, r * r, g * g, b * b, r * g, r * b, g * b]
    if degree >= 3:
        cols += [r ** 3, g ** 3, b ** 3, r * r * g, r * r * b, g * g * r, g * g * b, b * b * r, b * b * g, r * g * b]
    return np.stack(cols, axis=1)


def _fit_poly(s: np.ndarray, r: np.ndarray, degree: int, ridge: float) -> np.ndarray:
    keep = np.ones(len(s), bool)
    for _ in range(2):  # the refit drops the worst 5% of pixels (edges, noise, slight motion)
        p = _poly(s[keep], degree)
        reg = np.diag([0, 0, 0, 0] + [ridge] * (p.shape[1] - 4)) * len(p)
        beta = np.linalg.solve(p.T @ p + reg, p.T @ r[keep])
        err = np.abs(_poly(s, degree) @ beta - r).sum(axis=1)
        keep = err <= np.quantile(err, 0.95)
    return beta


_CUBE = np.stack(np.meshgrid(*[np.linspace(0, 1, 9)] * 3, indexing="ij"), -1).reshape(-1, 3)


def _fit_exact(s: np.ndarray, r: np.ndarray, ridge: float = 1e-4, degree: int = 3) -> dict:
    """Least-squares colour mapping from corresponding pixels (the handoff frame appears
    in both clips): a cubic polynomial in R, G, B, so each tone and hue can shift its own
    way (shadows cooler, highlights warmer...). On chains with known ground truth this
    beat distribution matching (HM-MVGD-HM) by 25-30%. Safety: colours the frame
    doesn't contain are extrapolated, so if the curve strays far from a plain linear
    fit anywhere in the colour cube, a gentler quadratic (then linear) fit is used."""
    s, r = s.astype(np.float64), r.astype(np.float64)
    p1 =np.linalg.lstsq(np.c_[np.ones(len(s)), s], r, rcond=None)[0]
    base = np.c_[np.ones(len(_CUBE)), _CUBE] @ p1
    for deg in range(degree, 1, -1):
        beta = _fit_poly(s, r, deg, ridge)
        if np.abs(np.clip(_poly(_CUBE, deg) @ beta, 0, 1) - np.clip(base, 0, 1)).max() <= 0.25:
            return {"kind": "poly", "degree": deg, "beta": beta.tolist()}
    beta =np.zeros((10, 3))
    beta[:4] = p1
    return {"kind": "poly", "degree": 2, "beta": beta.tolist()}


def _aligned(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation of two frames' brightness patterns (1 = same picture)."""
    za, zb = _z(a[None])[0].ravel(), _z(b[None])[0].ravel()
    return float(za @ zb / za.size)


def _pair_pixels(src: np.ndarray, tgt: np.ndarray, width: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Corresponding pixels of two same-picture frame stacks, lightly blurred by downsizing."""
    def shrink(f):
        h = max(2, int(round(width * f.shape[1] / f.shape[2])))
        return np.stack([np.asarray(Image.fromarray(x).resize((width, h), Image.BOX)) for x in f])
    a, b = shrink(src), shrink(tgt)
    return a.reshape(-1, 3).astype(np.float32) / 255.0, b.reshape(-1, 3).astype(np.float32) / 255.0


# ---- Oklab: split a correction into its lightness and colour parts ----

_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]], np.float32)
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]], np.float32)
_M1i, _M2i = np.linalg.inv(_M1).astype(np.float32), np.linalg.inv(_M2).astype(np.float32)


def to_oklab(rgb: np.ndarray) -> np.ndarray:
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return np.cbrt(lin @ _M1.T) @ _M2.T


def from_oklab(lab: np.ndarray) -> np.ndarray:
    lin = np.clip(((lab @ _M2i.T) ** 3) @ _M1i.T, 0, None)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)


def strengths(settings: dict) -> tuple[float, float]:
    """(brightness, colour) strengths; older sessions had one `strength` for both."""
    s = settings.get("strength", 1.0)
    return (float(settings.get("luma_strength", s)), float(settings.get("color_strength", s)))


def apply_image(stages: list[dict], img: np.ndarray, strength=1.0) -> np.ndarray:
    """uint8 HxWx3 -> corrected uint8 HxWx3. `strength` is one number (blend toward the
    original) or (brightness, colour): the correction is split in Oklab so lightness and
    colour can be applied by different amounts."""
    h, w, _ = img.shape
    x = img.reshape(-1, 3).astype(np.float32) / 255.0
    y = np.clip(apply_pixels(stages, x), 0, 1)
    lum, col = (strength, strength) if np.isscalar(strength) else strength
    if lum == col:
        if lum != 1.0:
            y = x + (y - x) * lum
    else:
        # lightness moves by `lum`; colour is (a, b) relative to lightness, which is what reads
        # as saturation, so brightening with colour at 0 keeps the look instead of washing out
        lx, ly = to_oklab(x), to_oklab(y)
        L = lx[:, :1] + (ly[:, :1] - lx[:, :1]) * lum
        sx, sy = lx[:, 1:] / (lx[:, :1] + 1e-3), ly[:, 1:] / (ly[:, :1] + 1e-3)
        y = from_oklab(np.concatenate([L, (sx + (sy - sx) * col) * (L + 1e-3)], axis=1))
    return np.clip(np.rint(y * 255), 0, 255).astype(np.uint8).reshape(h, w, 3)


def _pixels(frames: np.ndarray, limit: int = 400_000) -> np.ndarray:
    """Pool frames (n, h, w, 3) uint8 into (N, 3) float pixels, evenly subsampled."""
    x = frames.reshape(-1, 3)
    if len(x) > limit:
        x = x[:: len(x) // limit + 1]
    return x.astype(np.float32) / 255.0


# ==== color-exact decode / encode ==============================================================

def color_in(info: dict) -> str:
    """scale-filter options that decode this video's YUV with its own matrix and range."""
    cs = (info.get("color_space") or "").lower()
    matrix = {"bt709": "bt709", "smpte170m": "smpte170m", "bt470bg": "bt470", "bt2020nc": "bt2020",
              "bt2020c": "bt2020", "smpte240m": "smpte240m", "fcc": "fcc"}.get(cs)
    if not matrix:  # untagged: HD is almost always 709, SD 601
        matrix = "bt709" if (info.get("height") or 0) >= 720 else "smpte170m"
    pf = info.get("pix_fmt") or ""
    rng = info.get("color_range")
    if rng not in ("tv", "pc"):
        rng = "pc" if pf.startswith("yuvj") else "tv"
    return f"in_color_matrix={matrix}:in_range={rng}"


def decode_small(path: str, info: dict, width: int = SMALL_W, target: dict | None = None) -> np.ndarray:
    """Every frame as (n, h, w, 3) uint8 RGB at analysis size, color-exact."""
    tw = target["width"] if target else info["width"]
    th = target["height"] if target else info["height"]
    h = max(2, int(round(width * th / tw / 2)) * 2)
    # one scale does both the resize and the YUV->RGB conversion, so the matrix/range apply
    vf = f"scale={width}:{h}:{color_in(info)}:flags=area+accurate_rnd+full_chroma_int,format=rgb24"
    if target and target.get("fps_frac") and info.get("fps_frac") != target["fps_frac"]:
        vf = f"fps={target['fps_frac']}," + vf
    proc = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-an", "-vf", vf, "-f", "rawvideo", "-"],
                          capture_output=True, **_NOWIN)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode(errors="replace").strip()[-300:])
    n = len(proc.stdout) // (width * h * 3)
    return np.frombuffer(proc.stdout[: n * width * h * 3], np.uint8).reshape(n, h, width, 3)


def extract_png(path: str, info: dict, out_dir: str, target: dict | None = None, job=None) -> int:
    """Full-resolution PNG frames, color-exact (lossless between decode and encode)."""
    os.makedirs(out_dir, exist_ok=True)
    tw = target["width"] if target else info["width"]
    th = target["height"] if target else info["height"]
    vf = f"scale={tw}:{th}:{color_in(info)}:flags=lanczos+accurate_rnd+full_chroma_int,format=rgb24"
    if target and target.get("fps_frac") and info.get("fps_frac") != target["fps_frac"]:
        vf = f"fps={target['fps_frac']}," + vf
    media.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-an", "-vf", vf,
               "-compression_level", "1", os.path.join(out_dir, "f%06d.png")],
              job=job, duration=info.get("duration") or 0)
    return len([f for f in os.listdir(out_dir) if f.endswith(".png")])


def encode(png_dir: str, fps_frac: str, out: str, lossless: bool = False, job=None, frames: int = 0,
           start: int = 1, count: int | None = None) -> None:
    """PNG sequence -> H.264 (BT.709 limited, tagged) or lossless FFV1 master."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", fps_frac, "-start_number", str(start),
           "-i", os.path.join(png_dir, "f%06d.png")]
    if count:
        cmd += ["-frames:v", str(count)]
    if lossless:
        cmd += ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "gbrp", "-g", "1"]
    else:
        cmd += ["-vf", "scale=out_color_matrix=bt709:out_range=tv:flags=lanczos+accurate_rnd+full_chroma_int,format=yuv420p",
                "-c:v", "libx264", "-preset", "slow", "-crf", "14",
                "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                "-color_range", "tv", "-movflags", "+faststart"]
    fps = float(eval_frac(fps_frac)) or 25
    media.run(cmd + [out], job=job, duration=(count or frames) / fps if (count or frames) else 0)


def eval_frac(s: str) -> float:
    a, _, b = (s or "0/1").partition("/")
    try:
        return float(a) / float(b or 1)
    except (ValueError, ZeroDivisionError):
        return 0.0


# ==== overlap detection ========================================================================

def _z(frames: np.ndarray) -> np.ndarray:
    """(n, h, w, 3) -> z-scored grayscale (n, 36, 64): brightness/contrast drift ignored."""
    g = frames.astype(np.float32).mean(axis=3)
    n, h, w = g.shape
    g = np.stack([np.asarray(Image.fromarray(x).resize((64, 36), Image.BOX), np.float32) for x in g])
    g -= g.mean(axis=(1, 2), keepdims=True)
    g /= g.std(axis=(1, 2), keepdims=True) + 1e-6
    return g


def detect_overlap(a: np.ndarray, b: np.ndarray, max_n: int = MAX_OVERLAP) -> dict:
    """How many of b's first frames repeat a's last frames. Returns {n, error, errors}."""
    za, zb = _z(a[-max_n:]), _z(b[:max_n])
    errors = {}
    for n in range(1, min(max_n, len(za), len(zb)) + 1):
        errors[n] = float(np.abs(za[-n:] - zb[:n]).mean())
    if not errors:
        return {"n": 0, "error": None, "errors": {}}
    best = min(errors, key=lambda n: (round(errors[n], 3), n))
    if errors[best] > OVERLAP_MAX_ERR:
        return {"n": 0, "error": errors[best], "errors": errors}
    return {"n": best, "error": errors[best], "errors": errors}


# ==== analysis ===================================================================================

def _stats_series(frames: np.ndarray) -> tuple[list, list]:
    f = frames.astype(np.float64)
    y = (f @ np.array([0.2126, 0.7152, 0.0722])).mean(axis=(1, 2))
    mx, mn = f.max(axis=3), f.min(axis=3)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0).mean(axis=(1, 2))
    return [round(float(v), 2) for v in y], [round(float(v) * 100, 2) for v in sat]


def load_reference(path: str) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((SMALL_W * 2, SMALL_W * 2))
        return np.asarray(im)


def pick_fps(infos: list[dict], fps: str | None = "auto") -> str:
    """Output frame rate: 'auto' = the rate most segments already have (so the fewest
    clips are converted), 'first' = segment 1's, or an explicit rate like 16 or 30000/1001."""
    fracs = [i.get("fps_frac") for i in infos if i.get("fps_frac")]
    if not fracs:
        return "25/1"
    if fps in (None, "", "auto"):
        counts = {f: fracs.count(f) for f in fracs}
        return max(fracs, key=lambda f: (counts[f], -fracs.index(f)))
    if fps == "first":
        return fracs[0]
    v = eval_frac(str(fps))
    if not 1 <= v <= 240:
        raise ValueError("frame rate must be auto, first, or a number between 1 and 240")
    return str(fps) if "/" in str(fps) else f"{fps}/1"


def _fit_pair(src: np.ndarray, tgt: np.ndarray, method: str, aligned: bool) -> tuple[list[dict], str]:
    """Fit one segment. The exact method needs the two stacks to be the same picture;
    otherwise it falls back to HM-MVGD-HM, and the returned note says which ran."""
    if method == "exact":
        if aligned:
            a, b = _pair_pixels(src, tgt)
            return [_fit_exact(a, b)], "exact"
        return fit(_pixels(src), _pixels(tgt), "hm-mvgd-hm"), "hm-mvgd-hm (no matching frame to fit exactly)"
    return fit(_pixels(src), _pixels(tgt), method), method


def analyze(job, segments: list[str], reference: str | None = None, method: str = "exact",
            fit_frames: int = 5, mode: str = "seam", strength: float | None = None,
            overlaps: list[int | None] | None = None, names: list[str] | None = None,
            session: str | None = None, ids: list[str] | None = None, fps: str | None = "auto",
            luma_strength: float | None = None, color_strength: float | None = None) -> dict:
    """Probe, check, detect overlaps and fit one transform per segment (at analysis size)."""
    if len(segments) < 1:
        raise ValueError("add at least one segment")
    if method not in METHODS or mode not in MODES:
        raise ValueError("unknown method or mode")
    fit_frames = int(fit_frames)
    if not 1 <= fit_frames <= 60:
        raise ValueError("fit frames must be between 1 and 60")
    base = 1.0 if strength is None else float(strength)
    lum = base if luma_strength is None else float(luma_strength)
    col = base if color_strength is None else float(color_strength)
    if not (0 <= lum <= 1 and 0 <= col <= 1):
        raise ValueError("strengths must be between 0 and 1")
    infos = [media.probe(p) for p in segments]
    first = infos[0]
    target = {"width": first["width"], "height": first["height"], "fps_frac": pick_fps(infos, fps)}
    out_fps = eval_frac(target["fps_frac"])
    warnings = []
    for k, i in enumerate(infos, 1):
        if k > 1 and (i["width"], i["height"]) != (target["width"], target["height"]):
            warnings.append(f"segment {k} is {i['width']}x{i['height']}; it will be scaled to "
                            f"{target['width']}x{target['height']}")
        if i.get("fps_frac") and i["fps_frac"] != target["fps_frac"]:
            how = "frames repeated, motion may stutter" if eval_frac(i["fps_frac"]) < out_fps else "frames dropped"
            warnings.append(f"segment {k} runs at {i['fps']:g} fps; it will be converted to {out_fps:g} fps ({how})")
    if job:
        job.update(message="reading segments", progress=0, total=len(segments))
    small = []
    for k, (p, i) in enumerate(zip(segments, infos)):
        if job:
            job.check()
        small.append(decode_small(p, i, target=target))
        if job:
            job.update(progress=k + 1)
    if any(len(s) == 0 for s in small):
        raise ValueError("a segment decoded to no frames")

    # overlaps
    joins = []
    for k in range(1, len(small)):
        det = detect_overlap(small[k - 1], small[k])
        forced = overlaps[k] if overlaps and k < len(overlaps) and overlaps[k] is not None else None
        n = int(forced) if forced is not None else det["n"]
        n = max(0, min(n, len(small[k]) - 1))
        joins.append({"join": k, "detected": det["n"], "error": det["error"], "used": n, "forced": forced is not None})
    ovl = [0] + [j["used"] for j in joins]

    # reference, also at the analysis frame size so the exact fit can compare it pixel for pixel
    h0, w0 = small[0].shape[1:3]
    ref_img = load_reference(reference) if reference else small[0][0]
    ref_frame = np.asarray(Image.fromarray(ref_img).resize((w0, h0), Image.BOX))

    # one transform per segment
    transforms, fits, corrected_small = [], [], []
    for k, fr in enumerate(small):
        if job:
            job.check()
        if k == 0 or mode == "original":
            if method == "exact":  # the first frame was generated from the reference image
                src, tgt = fr[:1], ref_frame[None]
            else:
                src, tgt = fr[:fit_frames], ref_img[None]
            aligned = _aligned(fr[0], ref_frame) >= 0.9
        else:
            n = ovl[k]
            prev = corrected_small[k - 1]
            if n > 0:  # identical moment: pure drift
                src, tgt = fr[:n], prev[-n:]
                aligned = _aligned(src[0], tgt[0]) >= 0.9
            else:      # no overlap: neighbouring moments are the next best thing
                m = min(fit_frames, len(fr), len(prev))
                src, tgt, aligned = fr[:m], prev[-m:], False
        stages, note = _fit_pair(src, tgt, method, aligned)
        transforms.append(stages)
        fits.append(note)
        corrected_small.append(np.stack([apply_image(stages, f, (lum, col)) for f in fr]))

    # timeline stats (joined order, overlaps dropped) and each segment's average colour
    before_y, before_s, after_y, after_s, bounds, rgb_b, rgb_a = [], [], [], [], [], [], []
    for k, (fr, cf) in enumerate(zip(small, corrected_small)):
        bounds.append(len(before_y))
        by, bs = _stats_series(fr[ovl[k]:])
        ay, as_ = _stats_series(cf[ovl[k]:])
        before_y += by
        before_s += bs
        after_y += ay
        after_s += as_
        rgb_b.append([round(float(v), 1) for v in fr[ovl[k]:].reshape(-1, 3).mean(axis=0)])
        rgb_a.append([round(float(v), 1) for v in cf[ovl[k]:].reshape(-1, 3).mean(axis=0)])
    ry, rs = _stats_series(ref_img[None])

    sid = session or (time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2))
    d = _root(sid)
    os.makedirs(d, exist_ok=True)
    if reference and os.path.abspath(reference) != os.path.abspath(os.path.join(d, "reference.png")):
        with Image.open(reference) as im:
            im.convert("RGB").save(os.path.join(d, "reference.png"))
    s = {
        "id": sid, "created": time.time(),
        "segments": [{"path": p, "name": (names[k] if names else os.path.basename(p)),
                      "id": ids[k] if ids else None,
                      "frames": int(len(small[k])), "fps": infos[k]["fps"], "fps_frac": infos[k].get("fps_frac"),
                      "width": infos[k]["width"], "height": infos[k]["height"], "overlap": ovl[k], "fit": fits[k]}
                     for k, p in enumerate(segments)],
        "target": target, "joins": joins, "warnings": warnings,
        "settings": {"method": method, "fit_frames": fit_frames, "mode": mode,
                     "luma_strength": lum, "color_strength": col, "fps": fps or "auto",
                     "reference": "image" if reference else "first frame of segment 1",
                     "reference_token": os.path.splitext(os.path.basename(reference))[0]
                     if reference and os.path.basename(os.path.dirname(reference)) == "_refs" else None,
                     "overlaps": overlaps},
        "transforms": transforms,
        "stats": {"before_luma": before_y, "before_sat": before_s, "after_luma": after_y, "after_sat": after_s,
                  "bounds": bounds, "ref_luma": ry[0], "ref_sat": rs[0], "fps": out_fps,
                  "rgb_before": rgb_b, "rgb_after": rgb_a,
                  "rgb_ref": [round(float(v), 1) for v in ref_img.reshape(-1, 3).mean(axis=0)]},
        "outputs": None,  # any earlier render used the old transforms
    }
    _save(s)
    if job:
        job.update(message="analysed", progress=len(segments), total=len(segments))
    return {"session": sid, "segments": len(segments), "joins": [j["used"] for j in joins],
            "warnings": warnings, "fits": fits}


def _save(s: dict) -> None:
    with open(_root(s["id"], "session.json"), "w", encoding="utf-8") as f:
        json.dump(s, f)


def load(sid: str) -> dict:
    if not SID_RE.match(sid):
        raise ValueError("bad session id")
    p = _root(sid, "session.json")
    if not os.path.exists(p):
        raise LookupError("no such color match session")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def list_sessions() -> list[dict]:
    out = []
    base = _root()
    if os.path.isdir(base):
        for sid in sorted(os.listdir(base), reverse=True):
            if SID_RE.match(sid) and os.path.exists(_root(sid, "session.json")):
                s = load(sid)
                out.append({"id": sid, "created": s["created"], "segments": len(s["segments"]),
                            "names": [x["name"] for x in s["segments"]][:4],
                            "rendered": bool(s.get("outputs")), "method": s["settings"]["method"],
                            "mode": s["settings"]["mode"]})
    return out


def delete(sid: str) -> None:
    load(sid)
    shutil.rmtree(_root(sid), ignore_errors=True)


# ==== preview and render =======================================================================

def _grab(path: str, info: dict, index: int, target: dict | None) -> np.ndarray:
    """One full-resolution frame by index (counted at the output frame rate), color-exact."""
    fps = eval_frac(target["fps_frac"]) if target and target.get("fps_frac") else (info.get("fps") or 25)
    t = max(0.0, index / fps)
    tw = target["width"] if target else info["width"]
    th = target["height"] if target else info["height"]
    vf = f"scale={tw}:{th}:{color_in(info)}:flags=lanczos+accurate_rnd+full_chroma_int,format=rgb24"
    for tt in (t, max(0.0, t - 1 / fps)):
        proc = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{tt:.4f}", "-i", path, "-frames:v", "1",
                               "-an", "-vf", vf, "-f", "rawvideo", "-"], capture_output=True, **_NOWIN)
        if proc.returncode == 0 and len(proc.stdout) >= tw * th * 3:
            return np.frombuffer(proc.stdout[: tw * th * 3], np.uint8).reshape(th, tw, 3)
    raise RuntimeError("could not read that frame")


def preview(sid: str, seg: int, index: int, max_w: int = 640) -> bytes:
    """Side-by-side PNG: frame as generated | corrected."""
    s = load(sid)
    if not 0 <= seg < len(s["segments"]):
        raise ValueError("no such segment")
    sg = s["segments"][seg]
    info = media.probe(sg["path"])
    index = max(0, min(int(index), sg["frames"] - 1))
    img = _grab(sg["path"], info, index, s["target"])
    out = apply_image(s["transforms"][seg], img, strengths(s["settings"]))
    pair = Image.new("RGB", (img.shape[1] * 2 + 8, img.shape[0]), (20, 22, 26))
    pair.paste(Image.fromarray(img), (0, 0))
    pair.paste(Image.fromarray(out), (img.shape[1] + 8, 0))
    if pair.width > max_w * 2:
        pair.thumbnail((max_w * 2, max_w * 2))
    buf = io.BytesIO()
    pair.save(buf, "PNG")
    return buf.getvalue()


def render(job, sid: str, lossless: bool = False, crossfade: int = 0, keep_frames: bool = False,
           segment_clips: bool = False) -> dict:
    """Full-resolution pass: PNG in, correct, drop overlaps, optional crossfade, encode once."""
    s = load(sid)
    d = _root(sid)
    crossfade = max(0, min(int(crossfade), 12))
    frames_dir = os.path.join(d, "frames")
    shutil.rmtree(frames_dir, ignore_errors=True)
    os.makedirs(frames_dir)
    handoff_dir = os.path.join(d, "handoff")
    shutil.rmtree(handoff_dir, ignore_errors=True)
    os.makedirs(handoff_dir)
    strength = strengths(s["settings"])
    total = sum(sg["frames"] for sg in s["segments"])
    done, out_n = 0, 0
    ranges = []
    tail: list[np.ndarray] = []   # last `crossfade` corrected frames, held back for blending
    for k, sg in enumerate(s["segments"]):
        if job:
            job.check()
            job.update(message=f"segment {k + 1} of {len(s['segments'])}: extracting PNG")
        raw = os.path.join(d, "raw")
        shutil.rmtree(raw, ignore_errors=True)
        info = media.probe(sg["path"])
        n = extract_png(sg["path"], info, raw, s["target"])
        files = sorted(f for f in os.listdir(raw) if f.endswith(".png"))
        skip = sg["overlap"] if k > 0 else 0
        seg_start = out_n + 1
        last = None
        blend_left = len(tail) if k > 0 else 0
        for i, f in enumerate(files):
            if job and i % 10 == 0:
                job.check()
                job.update(message=f"segment {k + 1} of {len(s['segments'])}: correcting", progress=done + i, total=total)
            with Image.open(os.path.join(raw, f)) as im:
                arr = np.asarray(im.convert("RGB"))
            fixed = apply_image(s["transforms"][k], arr, strength)
            last = fixed
            if i < skip:
                continue  # repeated handoff frame: keep the previous segment's copy
            if blend_left:  # crossfade: blend held-back tail of the previous segment into this one
                j = len(tail) - blend_left
                a = (j + 1) / (len(tail) + 1)
                mixed = (tail[j].astype(np.float32) * (1 - a) + fixed.astype(np.float32) * a)
                fixed = np.clip(np.rint(mixed), 0, 255).astype(np.uint8)
                blend_left -= 1
                if blend_left == 0:
                    tail = []
            if crossfade and k < len(s["segments"]) - 1 and i >= len(files) - crossfade:
                tail.append(fixed)  # held back; written blended into the next segment
                continue
            out_n += 1
            Image.fromarray(fixed).save(os.path.join(frames_dir, f"f{out_n:06d}.png"), compress_level=1)
        ranges.append((seg_start, out_n))
        if last is not None:
            Image.fromarray(last).save(os.path.join(handoff_dir, f"segment_{k + 1:02d}_last_frame.png"))
        done += n
        shutil.rmtree(raw, ignore_errors=True)
    for fr in tail:  # nothing followed (shouldn't happen): write held frames as-is
        out_n += 1
        Image.fromarray(fr).save(os.path.join(frames_dir, f"f{out_n:06d}.png"), compress_level=1)

    fps_frac = s["target"]["fps_frac"] or "25/1"
    stem = os.path.splitext(s["segments"][0]["name"])[0]
    ext = "mkv" if lossless else "mp4"
    joined = f"{stem}-colormatched.{ext}"
    if job:
        job.update(message="encoding", progress=0, total=out_n)
    encode(frames_dir, fps_frac, os.path.join(d, joined), lossless=lossless, job=job, frames=out_n)
    clips = []
    if segment_clips:
        for k, (a, b) in enumerate(ranges):
            if b >= a:
                name = f"segment_{k + 1:02d}-colormatched.{ext}"
                encode(frames_dir, fps_frac, os.path.join(d, name), lossless=lossless, start=a, count=b - a + 1)
                clips.append(name)
    if not keep_frames:
        shutil.rmtree(frames_dir, ignore_errors=True)
    s = load(sid)
    s["outputs"] = {"joined": joined, "frames": out_n, "clips": clips, "lossless": lossless,
                    "crossfade": crossfade, "kept_frames": keep_frames,
                    "handoff": sorted(os.listdir(handoff_dir)), "rendered": time.time()}
    _save(s)
    return {"session": sid, "joined": joined, "frames": out_n, "clips": len(clips)}


def output_path(sid: str, name: str) -> str:
    load(sid)
    parts = name.replace("\\", "/").split("/")
    if any(p in ("", ".", "..") for p in parts) or len(parts) > 2 or (len(parts) == 2 and parts[0] not in ("handoff", "frames")):
        raise ValueError("bad file name")
    return os.path.join(_root(sid), *parts)


def correct_image(img_path: str, reference: str, out_path: str, method: str = "exact",
                  strength=1.0) -> str:
    """One-off: match a single image (e.g. a handoff frame) to the reference. The exact
    method is used when both show the same picture, HM-MVGD-HM otherwise."""
    with Image.open(img_path) as im:
        arr = np.asarray(im.convert("RGB"))
    small = np.asarray(Image.fromarray(arr).resize(
        (SMALL_W * 2, max(2, int(SMALL_W * 2 * arr.shape[0] / arr.shape[1]))), Image.BOX))
    ref = load_reference(reference)
    ref_r = np.asarray(Image.fromarray(ref).resize((small.shape[1], small.shape[0]), Image.BOX))
    if method == "exact":
        stages, _ = _fit_pair(small[None], ref_r[None], method, _aligned(small, ref_r) >= 0.9)
    else:
        stages = fit(_pixels(small[None]), _pixels(ref[None]), method)
    Image.fromarray(apply_image(stages, arr, strength)).save(out_path)
    return out_path
