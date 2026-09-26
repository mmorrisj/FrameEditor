"""Audio extraction: pull a video's audio track out as its own file.

  copy  the stream bit-for-bit in a matching container (AAC -> .m4a, Opus ->
        .opus, ...). No quality loss and near-instant.
  wav / flac / mp3  decode and convert.
"""
from __future__ import annotations

import glob
import os

from .. import library, media

FORMATS = {
    "copy": "Original stream (no re-encode)",
    "wav": "WAV (uncompressed PCM)",
    "flac": "FLAC (lossless)",
    "mp3": "MP3 (320 kbps)",
}

# codec -> container that can hold it without re-encoding
COPY_EXT = {
    "aac": "m4a", "alac": "m4a", "mp3": "mp3", "opus": "opus", "vorbis": "ogg",
    "flac": "flac", "ac3": "ac3", "eac3": "eac3", "dts": "dts", "truehd": "mka",
    "mp2": "mp2", "amr_nb": "amr", "amr_wb": "amr",
}


def _target(fmt: str, codec: str | None) -> tuple[str, list[str]]:
    if fmt == "copy":
        if codec and codec.startswith("pcm_"):
            return "wav", ["-c:a", "copy"]
        return COPY_EXT.get(codec or "", "mka"), ["-c:a", "copy"]
    if fmt == "wav":
        return "wav", ["-c:a", "pcm_s16le"]
    if fmt == "flac":
        return "flac", ["-c:a", "flac"]
    if fmt == "mp3":
        return "mp3", ["-c:a", "libmp3lame", "-b:a", "320k"]
    raise ValueError(f"unknown audio format {fmt!r}")


def extract(job, src: str, out_dir: str, fmt: str = "copy", stream: int = 0,
            info: dict | None = None) -> str:
    """Write audio stream `stream` (0 = first audio track) of `src` into out_dir."""
    info = info or media.probe(src)
    tracks = info.get("audio") or []
    if not tracks:
        raise ValueError("this video has no audio track")
    if not 0 <= stream < len(tracks):
        raise ValueError(f"audio track {stream} does not exist (video has {len(tracks)})")
    ext, codec_args = _target(fmt, tracks[stream]["codec"])
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(src))[0]
    suffix = f".track{stream + 1}" if len(tracks) > 1 else ""
    out = os.path.join(out_dir, f"{stem}{suffix}.{ext}")
    if job:
        job.update(message=f"extracting audio ({fmt})")
    tmp = out + ".part." + ext  # ffmpeg picks the muxer from the extension
    try:
        media.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-map", f"0:a:{stream}",
                   "-vn", *codec_args, tmp], job=job, duration=info.get("duration") or 0)
        os.replace(tmp, out)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return out


def out_dir(vid: str) -> str:
    return library.work_dir(vid, "audio")


def extract_video(job, vid: str, fmt: str = "copy", stream: int = 0) -> dict:
    v = library.require(vid)
    p = extract(job, v["path"], out_dir(vid), fmt=fmt, stream=stream, info=v["info"])
    return {"video_id": vid, "file": os.path.basename(p), "size": os.path.getsize(p)}


def extract_many(job, vids: list[str], fmt: str = "copy") -> dict:
    done, failed = [], {}
    job.update(total=len(vids), progress=0)
    for i, vid in enumerate(vids, 1):
        job.check()
        try:
            v = library.require(vid)
            job.update(message=v["name"])
            p = extract(None, v["path"], out_dir(vid), fmt=fmt, info=v["info"])
            done.append({"video_id": vid, "file": os.path.basename(p)})
        except Exception as e:
            failed[vid] = str(e)
        job.update(progress=i)
    return {"done": done, "failed": failed}


def list_outputs(vid: str) -> list[dict]:
    d = out_dir(vid)
    out = []
    for p in glob.glob(os.path.join(d, "*")):
        if os.path.isfile(p) and ".part." not in p:
            out.append({"file": os.path.basename(p), "size": os.path.getsize(p),
                        "created": os.path.getmtime(p)})
    return sorted(out, key=lambda d: d["created"], reverse=True)
