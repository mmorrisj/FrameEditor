# FrameKit

A suite of video tools for a folder full of videos, built on ffmpeg. Each tool
works on its own, from the web app or the command line:

| Tool | What it does |
| --- | --- |
| **Frames** | Break a video into images: every frame, every Nth frame, N per second, scene changes only, or keyframes only. Every frame's source timestamp is recorded. |
| **Resize** | Scale videos without ever stretching the picture: by percent, to a height or width, or to an exact size. For a different shape, such as vertical or square, choose bars, a blurred background, or crop to fill, and preview a frame first. Non-square pixels and phone rotation are corrected, and audio is copied untouched. |
| **Reverse** | Make a rewound copy that plays backwards, optionally faster or as a boomerang (forwards, then rewind). Audio can be reversed, kept playing forwards, or dropped. Long and high-resolution videos are reversed in chunks, so memory use stays flat. |
| **Audio** | Pull the audio track out untouched (AAC stays AAC in an .m4a), or convert to WAV, FLAC or MP3. |
| **Duplicates** | Find exact copies, re-encodes (other resolution, bitrate or container) and trimmed copies. Extras go to a quarantine folder that can be undone. |
| **Groups** | Cluster videos by how similar their sampled frames look. Adjust the cutoff with a slider, drag videos between groups, export, or file them into folders. |
| **Unique frames** | A drop folder for images. Near-duplicates are moved out automatically, keeping the sharpest image of each look-alike group, so what's left is mostly distinct angles and moments. |
| **Frame analysis** | Within one set of extracted frames: find shots, flag duplicate frames, cluster frames, and order them by time, cluster, similarity chain or by hand. Export zips, a contact sheet or a re-rendered video. |
| **Frame editor** | The original FrameEditor: blur, pixelate or clone away a region across a range of frames, then repackage with the original audio copied back bit for bit. |

## Setup on Windows

Needs Python 3.10+ and ffmpeg. Install ffmpeg with `winget install Gyan.FFmpeg`,
then open a new terminal so it is on PATH. In PowerShell, from this folder:

```powershell
python -m venv venv
venv\Scripts\pip install -r requirements.txt
venv\Scripts\python -m framekit roots add D:\Videos      # repeat per folder
.\start-framekit.bat                                     # opens http://localhost:8082
```

To have it start in the background whenever you sign in (no admin needed,
output goes to `work\server.log`):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install-autostart.ps1 -StartNow
powershell -ExecutionPolicy Bypass -File scripts\uninstall-autostart.ps1   # undo
```

The app only listens on this machine by default. It has no login, so reach it
from other machines only on a network you trust. To do that, set
`FRAMEKIT_HOST=0.0.0.0` before running `start-framekit.bat`, or pass
`-ListenHost 0.0.0.0` to the installer. Windows Firewall asks to allow Python
the first time.

Two Windows specifics:

- A file that is open elsewhere, for example playing in a video player or open
  in an image viewer, can't be moved. The quarantine skips it and reports it.
- Very deep folder trees can pass Windows' 260-character path limit inside the
  quarantine. Turning on long path support in Windows avoids that. The setting
  is `LongPathsEnabled` under `HKLM\SYSTEM\CurrentControlSet\Control\FileSystem`.

## Setup on Linux or macOS

```bash
python -m venv venv && venv/bin/pip install -r requirements.txt
venv/bin/python -m framekit roots add /path/to/videos
venv/bin/python -m framekit serve                       # http://localhost:8082
```

Optional extras:

- **CLIP grouping** groups by content ("a red car on a beach") instead of look.
  Install with `pip install -r requirements-clip.txt`. The model (about 600 MB)
  downloads on first use and uses a GPU when one is available.
- **Audio fingerprints** in duplicate detection catch copies that were cropped or
  overlaid. They need Chromaprint's `fpcalc` on PATH. On Windows, download it
  from acoustid.org/chromaprint and put `fpcalc.exe` in a folder on PATH.

## Library folders

The web app works on the videos inside the configured folders, listed in
`framekit.json` (see `framekit.example.json`) or managed with
`python -m framekit roots add|remove PATH`. Folders are set server-side only,
so the browser can never point the app elsewhere on disk. Uploads from the web
app land in `work/uploads`, which is always part of the library.

Everything the suite generates lives under `work/` (override with
`FRAMEKIT_WORK`): the index database, cached fingerprints, extracted frames and
audio, exports, and the quarantine. It is gitignored and safe to delete. Your
video files are only ever touched by an explicit quarantine or "move into
group folders", and both can be undone.

## Unique frames drop folder

Drop images on the Unique frames page, or copy them into `work\frames-inbox`.
To use another folder, add `"unique": {"inbox": "D:/Frames"}` to `framekit.json`.
Each subfolder is its own set, and sets are never compared with each other.

With watching on, a set is processed a few seconds after it stops changing,
so a large copy is never processed half-finished. Within a set, images are
ranked sharpest first. An image is removed when it is within the similarity
cutoff of an image already kept. The default cutoff of 0.10 removes frames of
nearly the same moment and keeps distinct views. Raise it to keep fewer, lower
it to keep more, and use Preview to see the look-alike pairs before anything
moves.

Removed images go to the quarantine. Undo puts them back and pins them, so
automatic passes never remove them again.

## Command line

Folder arguments are indexed on the fly, so none of these need library setup.

```bash
python -m framekit frames clip.mp4 --mode scene --value 0.3     # or all | nth | fps | keyframes
python -m framekit audio ~/Videos --format copy                 # every video in a folder
python -m framekit resize D:/Videos --preset 720p              # or --size 1080x1920 --fit blur
python -m framekit reverse clip.mp4 --speed 2                   # or --boomerang, --audio keep|none
python -m framekit dupes ~/Videos --deep                        # dry run, prints what it found
python -m framekit dupes ~/Videos --apply                       # quarantine every extra
python -m framekit quarantine list | undo BATCH | purge BATCH --yes
python -m framekit group ~/Videos --export groups.csv           # --backend clip, --threshold 0.25
python -m framekit unique D:/Frames/shoot1                  # dry run: lists look-alike pairs
python -m framekit unique D:/Frames/shoot1 --apply --threshold 0.15
python -m framekit unique --watch                           # watch the inbox without the web app
python -m framekit analyze clip.mp4 --mode fps --value 2 --order chain --export ordered sheet video
```

## How the matching works

- **Samples.** Each video gets 12 frames sampled evenly between 5% and 95% of its
  length, skipping intros and outros. They are decoded with one accurate seek
  each and cached until the file changes. Duplicates and groups share them.
- **Exact duplicates** have the same size and SHA-256.
- **Re-encodes** have durations within 2% (at least one second) and sampled frames
  whose 64-bit perceptual hashes average within 10 bits. Blank frames are ignored.
- **Trimmed copies** come from the deep scan. It hashes every video at 1 fps and
  slides the shorter sequence along the longer one. Trims are reported but never
  selected for removal, since the trim may be the version you want.
- **Keeper choice** prefers the highest resolution, then bitrate, then length,
  then the earlier-listed library folder, then the oldest file.
- **Grouping** compares videos set to set. Each sampled frame of one video is
  matched to its closest frame in the other and the scores are averaged both
  ways. Average-linkage clustering is then cut at the chosen distance, so the
  number of groups is never guessed.

## Linux service

`frameeditor.service` is a systemd unit running the Flask server behind a
reverse proxy (single-user home LAN use). `app.py` is its entry point.

```bash
sudo cp frameeditor.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now frameeditor
```

Editor projects created before the suite existed stay in `projects/` and keep
working. New ones go to `work/editor/`.

## Layout

- `framekit/` holds the shared core:
  - `config.py`: work folder and library roots.
  - `db.py`: SQLite index, feature cache and file-move log.
  - `jobs.py`: background jobs with progress and cancel.
  - `media.py`: ffmpeg and ffprobe helpers.
  - `library.py`: scanning and indexing.
  - `samples.py`: cached per-video fingerprints.
  - `features.py`: perceptual hash and embeddings.
  - `fileops.py`: logged moves, quarantine, undo and purge.
- `framekit/tools/` has one module per tool: `frames`, `audio`, `resize`, `reverse`, `dupes`,
  `grouping`, `unique`, `analysis`, and `editor/`.
- `start-framekit.bat` and `scripts/` hold the Windows launcher and the
  start-at-sign-in installer.
- `framekit/web/` is the Flask app. There is one template per page, with shared
  `static/app.js` and `app.css` and no build step.
- `framekit/cli.py` is the command line.
