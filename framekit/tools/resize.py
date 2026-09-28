"""Resize videos without ever stretching or squashing the picture.

Target, one of:
  percent        scale both sides, e.g. 50
  width / height one side; the other follows the video's shape
  width + height an exact frame size (a "box"), e.g. 1080x1920 for vertical

When a box has a different shape from the video, `fit` decides what happens:
  pad    whole picture, bars fill the rest (black)
  blur   whole picture, over a blurred, zoomed copy of itself
  crop   fill the box, trimming the overflow (`anchor` picks which part stays)
  inside whole picture, scaled to fit within the box; output keeps the
         video's own shape (no bars, no crop)

The picture is first corrected to square pixels (anamorphic video) and upright
orientation (phone video stored sideways with a rotation flag), so the result
looks the way the original plays. Audio is copied untouched when the MP4
container can hold it, otherwise converted to AAC.
"""
from __future__ import annotations

import io
import os
import shutil
import subprocess

from PIL import Image

from .. import config, library, media

FITS = {"pad": "Add bars", "blur": "Blurred background", "crop": "Crop to fill",
        "inside": "Fit inside (keep original shape)"}
ANCHORS = ("center", "top", "bottom", "left", "right")
QUALITY = {"high": 18, "balanced": 21, "small": 25}
PRESETS = {  # name -> (label, width, height, percent)
    "50pct": ("Half size (50%)", None, None, 50),
    "2160p": ("4K (2160 tall)", None, 2160, None),
    "1080p": ("1080p (1080 tall)", None, 1080, None),
    "720p": ("720p (720 tall)", None, 720, None),
    "480p": ("480p (480 tall)", None, 480, None),
    "vertical": ("Vertical 1080×1920", 1080, 1920, None),
    "square": ("Square 1080×1080", 1080, 1080, None),
    "portrait45": ("Portrait 4:5 1080×1350", 1080, 1350, None),
    "wide1080": ("Widescreen 1920×1080", 1920, 1080, None),
}
MP4_AUDIO = {"aac", "mp3", "ac3", "eac3", "opus", "alac", "flac"}


def display_size(info: dict) -> tuple[int, int]:
    """Width and height as the video is actually shown (pixel shape and rotation applied)."""
    w = round(info["width"] * (info.get("sar") or 1.0))
    h = info["height"]
    return (h, w) if info.get("rotation") in (90, 270) else (w, h)


def _even(n) -> int:
    n = int(round(float(n)))
    if n < 16 or n > 8192:
        raise ValueError("sizes must be between 16 and 8192 pixels")
    return n - n % 2


def resolve(preset: str | None = None, width=None, height=None, percent=None) -> dict:
    if preset:
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}")
        _, width, height, percent = PRESETS[preset]
    width = _even(width) if width not in (None, "") else None
    height = _even(height) if height not in (None, "") else None
    percent = float(percent) if percent not in (None, "") else None
    if percent is not None and not 5 <= percent <= 400:
        raise ValueError("percent must be between 5 and 400")
    if not (percent or width or height):
        raise ValueError("give a percent, a width, a height, or both")
    return {"width": width, "height": height, "percent": percent}


def build_filter(info: dict, width=None, height=None, percent=None, fit: str = "pad",
                 anchor: str = "center") -> tuple[str, str]:
    """(filter_complex graph ending in [v], short description for file names)."""
    if fit not in FITS:
        raise ValueError(f"fit must be one of {list(FITS)}")
    if anchor not in ANCHORS:
        raise ValueError(f"anchor must be one of {ANCHORS}")
    L = ":flags=lanczos"
    # ffmpeg applies the rotation flag while decoding; this fixes non-square pixels
    pre = f"scale=trunc(iw*sar/2)*2:ih{L},setsar=1," if abs((info.get("sar") or 1) - 1) > 0.01 else ""
    if percent:
        p = percent / 100
        return f"[0:v]{pre}scale=trunc(iw*{p:g}/2)*2:trunc(ih*{p:g}/2)*2{L},setsar=1[v]", f"{percent:g}pct"
    if width and not height:
        return f"[0:v]{pre}scale={width}:-2{L},setsar=1[v]", f"w{width}"
    if height and not width:
        return f"[0:v]{pre}scale=-2:{height}{L},setsar=1[v]", f"{height}p"
    W, H = width, height
    fit_in = f"scale={W}:{H}:force_original_aspect_ratio=decrease:force_divisible_by=2{L}"
    cover = f"scale={W}:{H}:force_original_aspect_ratio=increase{L}"
    tag = f"{W}x{H}-{fit}"
    if fit == "inside":
        return f"[0:v]{pre}{fit_in},setsar=1[v]", f"in{W}x{H}"
    if fit == "pad":
        return f"[0:v]{pre}{fit_in},pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1[v]", tag
    if fit == "crop":
        x = {"left": "0", "right": "iw-ow"}.get(anchor, "(iw-ow)/2")
        y = {"top": "0", "bottom": "ih-oh"}.get(anchor, "(ih-oh)/2")
        return f"[0:v]{pre}{cover},crop={W}:{H}:{x}:{y},setsar=1[v]", tag + ("" if anchor == "center" else f"-{anchor}")
    # blur: the picture over a zoomed, blurred copy of itself
    return (f"[0:v]{pre}split[a][b];[a]{cover},crop={W}:{H},boxblur=luma_radius=min(h\\,w)/20:luma_power=2,"
            f"eq=brightness=-0.08[bg];[b]{fit_in}[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[v]"), tag


def preview(src: str, info: dict, t: float | None = None, max_side: int = 640, **opts) -> bytes:
    """PNG of one frame of the result, for checking the framing before encoding."""
    target = resolve(opts.get("preset"), opts.get("width"), opts.get("height"), opts.get("percent"))
    graph, _ = build_filter(info, **target, fit=opts.get("fit", "pad"), anchor=opts.get("anchor", "center"))
    t = (info.get("duration") or 0) * 0.3 if t is None else t
    graph = graph[:-3] + f",scale='min({max_side},iw)':-2[v]"
    proc = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", src, "-frames:v", "1",
                           "-filter_complex", graph, "-map", "[v]", "-f", "image2pipe", "-c:v", "png", "-"],
                          capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(proc.stderr.decode(errors="replace").strip()[-300:] or "preview failed")
    return proc.stdout


def resize(job, src: str, out: str, preset: str | None = None, width=None, height=None, percent=None,
           fit: str = "pad", anchor: str = "center", quality: str = "high", info: dict | None = None) -> dict:
    info = info or media.probe(src)
    target = resolve(preset, width, height, percent)
    graph, _ = build_filter(info, **target, fit=fit, anchor=anchor)
    if quality not in QUALITY:
        raise ValueError(f"quality must be one of {list(QUALITY)}")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-filter_complex", graph,
           "-map", "[v]", "-map", "0:a?", "-map_metadata", "0",
           "-c:v", "libx264", "-preset", "medium", "-crf", str(QUALITY[quality]), "-pix_fmt", "yuv420p"]
    codecs = {a["codec"] for a in info.get("audio") or []}
    cmd += ["-c:a", "copy"] if codecs and codecs <= MP4_AUDIO else ["-c:a", "aac", "-b:a", "192k"]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    part = out + ".part.mp4"
    if job:
        job.update(message="resizing")
    try:
        media.run(cmd + ["-movflags", "+faststart", part], job=job, duration=info.get("duration") or 0)
        os.replace(part, out)
    finally:
        if os.path.exists(part):
            os.remove(part)
    o = media.probe(out)
    return {"file": os.path.basename(out), "size": os.path.getsize(out),
            "from": list(display_size(info)), "to": list(display_size(o))}


# --- library videos -----------------------------------------------------------------------

def out_dir(vid: str) -> str:
    return library.work_dir(vid, "resized")


def output_name(src: str, info: dict, preset=None, width=None, height=None, percent=None,
                fit="pad", anchor="center") -> str:
    target = resolve(preset, width, height, percent)
    _, tag = build_filter(info, **target, fit=fit, anchor=anchor)
    return f"{os.path.splitext(os.path.basename(src))[0]}-{tag}.mp4"


def resize_video(job, vid: str, to_library: bool = False, **opts) -> dict:
    v = library.require(vid)
    name = output_name(v["path"], v["info"], **{k: opts.get(k) for k in
                                                 ("preset", "width", "height", "percent", "fit", "anchor")
                                                 if opts.get(k) is not None})
    out = os.path.join(out_dir(vid), name)
    res = resize(job, v["path"], out, **opts)  # fresh probe: pixel shape and rotation matter here
    res["video_id"] = vid
    if to_library:
        from .. import fileops
        dst = fileops._free(os.path.join(config.UPLOADS, name))
        shutil.copy2(out, dst)
        res["library_id"] = library.add_path(dst)["id"]
    return res


def list_outputs(vid: str) -> list[dict]:
    d = out_dir(vid)
    if not os.path.isdir(d):
        return []
    out = [{"file": f, "size": os.path.getsize(os.path.join(d, f)),
            "created": os.path.getmtime(os.path.join(d, f))}
           for f in os.listdir(d) if f.endswith(".mp4") and ".part." not in f]
    return sorted(out, key=lambda x: x["created"], reverse=True)


def preview_image(src: str, info: dict, **opts) -> Image.Image:
    return Image.open(io.BytesIO(preview(src, info, **opts)))
