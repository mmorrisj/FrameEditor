# FrameEditor

Fix a few bad frames in a video without losing anything else. Upload a video,
browse every frame, drag a box over the problem, blur / pixelate / clone it away
across a range of frames, then repackage — the original audio stream is copied
back bit-for-bit.

Built for the "AI video is great except three frames contain something that
shouldn't be there" problem.

## How it works

1. **Extract** — ffmpeg decodes the video once, writing every frame as a
   lossless PNG plus a 200px thumbnail for the filmstrip. Extraction is
   normalized to constant frame rate (AI tools sometimes emit VFR, which would
   otherwise drift the audio out of sync on re-encode).
2. **Edit** — in the browser: drag a rectangle on the frame, pick an operation,
   apply it to one frame or a range. Every frame's pristine copy is kept, so
   any frame can be reverted. For anything the built-in tools can't do,
   download the frame, edit it in GIMP, and upload the replacement.
   - **Blur** — Gaussian blur of the region
   - **Pixelate** — mosaic the region
   - **Clone** — copy the region from the previous frame; applied ascending
     over a range, a clean patch cascades through consecutive bad frames
3. **Repackage** — the PNG sequence is re-encoded (x264, crf 18, visually
   lossless) at the original frame rate and muxed with the untouched audio
   stream from the original file.

## Setup

Requires ffmpeg/ffprobe on PATH.

```bash
python -m venv venv && venv/bin/pip install -r requirements.txt
venv/bin/python app.py        # http://localhost:8082
```

Uploads, frames and outputs live under `projects/` (gitignored). A one-minute
1080p30 video is ~1,800 PNGs and a few GB while a project is open — delete
projects when done.

## Deployment

`frameeditor.service` is a systemd unit running the Flask dev server behind a
reverse proxy (single-user home LAN use):

```bash
sudo cp frameeditor.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now frameeditor
```

## Layout

- `video.py` — ffmpeg/ffprobe plumbing: probe, extract, repackage, job progress
- `edits.py` — Pillow region operations + pristine-copy backup/revert
- `app.py` — thin Flask layer over the two modules
- `templates/index.html` — the whole frontend (vanilla JS, no build step)
