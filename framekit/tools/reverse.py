"""Reverse a video so it plays rewound.

ffmpeg's `reverse` filter buffers every frame it reverses, so reversing a
whole video at once can need tens of gigabytes. Instead the video is cut into
chunks sized to stay near MEMORY_BUDGET, each chunk is reversed and encoded,
and the chunks are joined last-to-first without re-encoding again.

Audio is handled on its own track as uncompressed WAV chunks, reversed the
same way, then encoded once at the end. Chunking the AAC directly would leave
small gaps (encoder priming) at every join.

Options:
  audio      reverse (default) | keep (original audio, forwards) | none
  speed      playback speed of the result, e.g. 2 for a fast rewind
  boomerang  play forwards, then rewind back to the start
"""
from __future__ import annotations

import math
import os
from fractions import Fraction
import shutil
import tempfile

from .. import config, library, media

AUDIO = {"reverse": "Reversed", "keep": "Original (plays forwards)", "none": "No audio"}
MEMORY_BUDGET = 600 * 1024**2  # bytes of decoded frames held per chunk
MAX_CHUNK_S, MIN_CHUNK_S = 30.0, 1.0


def chunk_seconds(info: dict) -> float:
    frame_bytes = max(1, info["width"] * info["height"]) * 1.5  # yuv420p
    fps = info.get("fps") or 30
    return max(MIN_CHUNK_S, min(MAX_CHUNK_S, MEMORY_BUDGET / (frame_bytes * fps)))


def _atempo(speed: float) -> str:
    """atempo chain for any speed (each stage limited to 0.5..2 on older ffmpeg)."""
    stages, s = [], speed
    while s > 2.0:
        stages.append("atempo=2.0")
        s /= 2.0
    while s < 0.5:
        stages.append("atempo=0.5")
        s /= 0.5
    stages.append(f"atempo={s:.6g}")
    return ",".join(stages)


def reverse(job, src: str, out: str, audio: str = "reverse", speed: float = 1.0,
            boomerang: bool = False, crf: int = 18, info: dict | None = None,
            chunk_s: float | None = None) -> dict:
    if audio not in AUDIO:
        raise ValueError(f"audio must be one of {list(AUDIO)}")
    speed = float(speed)
    if not 0.25 <= speed <= 16:
        raise ValueError("speed must be between 0.25 and 16")
    info = info or media.probe(src)
    dur = float(info.get("duration") or 0)
    if dur <= 0:
        raise ValueError("could not determine the video's length")
    has_audio = info.get("has_audio") and audio != "none"

    # chunks are a whole number of frames and start exactly on the frame grid,
    # so no frame is repeated or dropped where two chunks meet
    fps = Fraction(info["fps_frac"]) if info.get("fps_frac") else Fraction(30)
    per = max(1, round((float(chunk_s) if chunk_s else chunk_seconds(info)) * fps))
    n = max(1, math.ceil(dur * fps / per))
    start = lambda i: f"{float(i * per / fps):.6f}"
    span = f"{float(per / fps):.6f}"
    lead = f"{float((per + 2) / fps):.6f}"  # decode a little extra; trim cuts it exactly
    fps_filter = f"fps={info['fps_frac']}," if info.get("fps_frac") else ""
    speed_v = f",setpts=PTS/{speed:g}" if speed != 1 else ""
    vbase = "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p"
    venc = ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(int(crf)),
            "-video_track_timescale", "90000", "-an"]
    steps = n + (1 if boomerang else 0) + (n if has_audio and audio == "reverse" else 0) \
        + (1 if has_audio and (audio == "keep" or boomerang) else 0) + 1
    done = 0

    def tick(msg):
        nonlocal done
        done += 1
        if job:
            job.check()
            job.update(message=msg, progress=done, total=steps)

    tmp = tempfile.mkdtemp(prefix="reverse-", dir=config.WORK)
    try:
        vparts, aparts = [], []
        if boomerang:  # the forward half, encoded with identical settings so it joins cleanly
            p = os.path.join(tmp, "fwd.mp4")
            media.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src,
                       "-vf", f"{fps_filter}{vbase}{speed_v}", *venc, p])
            vparts.append(p)
            tick("encoding the forward half")
        # video chunks, reversed, joined last-to-first
        for i in reversed(range(n)):
            p = os.path.join(tmp, f"v{i:05d}.mp4")
            try:
                media.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", start(i), "-t", lead, "-i", src,
                           "-vf", f"{fps_filter}trim=end_frame={per},{vbase},reverse{speed_v}", *venc, p])
            except RuntimeError:
                if i != n - 1:
                    raise
                # the length estimate can overshoot by a frame, leaving the last chunk empty
            if os.path.exists(p) and os.path.getsize(p) > 0:
                vparts.append(p)
            tick(f"reversing video {n - i}/{n}")

        if has_audio:
            atempo = f",{_atempo(speed)}" if speed != 1 else ""
            wav = ["-vn", "-c:a", "pcm_s16le"]
            if boomerang or audio == "keep":
                p = os.path.join(tmp, "afwd.wav")
                media.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-map", "0:a:0",
                           *(["-af", atempo[1:]] if atempo else []), *wav, p])
                fwd_audio = p
                tick("preparing audio")
            if boomerang:
                aparts.append(fwd_audio)
            if audio == "reverse":
                for i in reversed(range(n)):
                    p = os.path.join(tmp, f"a{i:05d}.wav")
                    media.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", start(i), "-t", span,
                               "-i", src, "-map", "0:a:0", "-af", f"areverse{atempo}", *wav, p])
                    if os.path.exists(p) and os.path.getsize(p) > 44:
                        aparts.append(p)
                    tick(f"reversing audio {n - i}/{n}")
            else:  # keep: the original audio plays forwards under the rewound picture
                aparts.append(fwd_audio)

        def concat_list(parts, name):
            lst = os.path.join(tmp, name)
            with open(lst, "w", encoding="utf-8") as f:
                for p in parts:
                    f.write("file '" + p.replace("\\", "/").replace("'", "'\\''") + "'\n")
            return lst

        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        part = out + ".part.mp4"
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-f", "concat", "-safe", "0", "-i", concat_list(vparts, "video.txt")]
        if aparts:
            cmd += ["-f", "concat", "-safe", "0", "-i", concat_list(aparts, "audio.txt"),
                    "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "192k", "-shortest"]
        cmd += ["-c:v", "copy", "-movflags", "+faststart", part]
        media.run(cmd)
        os.replace(part, out)
        tick("done")
        return {"file": os.path.basename(out), "size": os.path.getsize(out),
                "duration": round((dur * (2 if boomerang else 1)) / speed, 2)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        if os.path.exists(out + ".part.mp4"):
            os.remove(out + ".part.mp4")


# --- library videos -------------------------------------------------------------------

def out_dir(vid: str) -> str:
    return library.work_dir(vid, "reversed")


def output_name(src: str, audio: str, speed: float, boomerang: bool) -> str:
    stem = os.path.splitext(os.path.basename(src))[0]
    tags = ["boomerang" if boomerang else "reversed"]
    if speed != 1:
        tags.append(f"{speed:g}x")
    if audio != "reverse":
        tags.append({"keep": "orig-audio", "none": "silent"}[audio])
    return f"{stem}-{'-'.join(tags)}.mp4"


def reverse_video(job, vid: str, audio: str = "reverse", speed: float = 1.0,
                  boomerang: bool = False, to_library: bool = False) -> dict:
    v = library.require(vid)
    name = output_name(v["path"], audio, speed, boomerang)
    out = os.path.join(out_dir(vid), name)
    res = reverse(job, v["path"], out, audio=audio, speed=speed, boomerang=boomerang, info=v["info"])
    res["video_id"] = vid
    if to_library:  # a copy in the uploads folder, so every other tool can use it
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
