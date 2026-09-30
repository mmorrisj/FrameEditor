# FrameKit

A suite of video tools for a folder full of videos, built on ffmpeg. Each tool
works on its own, from the web app or the command line:

| Tool | What it does |
| --- | --- |
| **Frames** | Break a video into images: every frame, every Nth frame, N per second, scene changes only, or keyframes only. Every frame's source timestamp is recorded. |
| **Scenes** | Split a video into one clip per scene. Tune the sensitivity with a slider and see the cuts update instantly, merge scenes or add a cut at any point, then export frame-accurate clips (re-encoded) or fast ones (no re-encode, cuts snap to keyframes). |
| **Lineage** | Work out which AI clip continues which. Clips that start on another clip's last frame are its children, and clips that share a first frame are alternate takes, so whole generation trees are rebuilt from the pictures alone. Index saved folders (or any one folder on its own), browse chains generation by generation, hover a take to play it, click takes to choose your chain, and play the chosen chain as one continuous video. Fix missed links by hand (with picture-content suggestions for handoff frames that were edited), then send the chain to Color match to join it. A take can end before its last frame (e.g. when the face is hidden at the very end): set the end frame, nudge it a frame at a time with a live preview and face check, and save that frame as a handoff PNG. A clip generated from it links back automatically, even if you later move the end. |
| **Sounds** | Cut audio into clips and keep them in a sound library. Open any video with audio (or upload WAV/MP3/FLAC/...), see its waveform with dead gaps shaded and sound onsets marked, drag to select (snapping to onsets), and save the selection with a name, category and tags. Clips are trimmed to the sound, faded against clicks, optionally peak-normalised, and stored as 48 kHz / 24-bit WAV in one folder per category. "Found sounds" lists every sound between the gaps so a long recording can be split and saved in one go. The library is searchable by name, tag and category, with preview, favourites and a link back to each clip's source. |
| **Mix** | Layer sounds from the sound library onto a video: a Colour match render of a chain ("Layer sounds…" next to the rendered video, with the joins between segments as snap points) or any library video. Drag sounds onto tracks, move them between tracks, drag either edge to trim, cut with the ✂ tool or split at the playhead (S), duplicate (Ctrl+D), and set volume, fades and looping per clip; each track and the video's own audio have a volume and mute. Preview plays in sync with the video, and rendering mixes everything onto the video without re-encoding the picture. Cleanup per clip: match loudness (every clip to a common level), noise reduction (light, medium, strong), low cut, high cut, de-click and pan, processed once and cached so the preview plays exactly what renders. Ducking per track: dip a music or ambience track while the video's own audio (or another track) plays. Master: loudness target (streaming -14, web -16, broadcast -23 LUFS, checked after encoding so the file lands on it), a true-peak limiter with a chosen ceiling, and optional glue compression; Measure sets the preview to the level the render will have. |
| **Color match** | Fix the color drift that compounds across chained AI video segments (each one darker or more saturated than the last), then join them. One color transform is fitted per segment and applied to every frame, so nothing flickers and real lighting changes survive. By default each segment is matched to the corrected end of the one before it, at the repeated handoff frame, which is detected and dropped at each join. Frames go through PNG and the color conversion is pinned, so the tool adds no shift of its own. Methods: an exact fit from the repeated handoff frame (default; about 25-30% more accurate than distribution matching in tests), HM-MVGD-HM, MKL, HM-MKL-HM, MVGD and histogram matching. Brightness and colour corrections have separate strengths, and the output frame rate defaults to what most segments use. Also writes the corrected last frame of each segment for regenerating the next one, and can correct a single image. |
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
- **Face likeness** (Lineage) scores how closely each take's face matches the
  face in its chain's original start frame, so you can see which takes stayed on
  model and where a chain drifts. Install with
  `pip install --no-deps insightface==2.0` then `pip install -r requirements-faces.txt`.
  The InsightFace models (about 300 MB) download once, run locally (GPU when
  available), and are licensed for non-commercial use.
- **Audio fingerprints** in duplicate detection catch copies that were cropped or
  overlaid. They need Chromaprint's `fpcalc` on PATH. On Windows, download it
  from acoustid.org/chromaprint and put `fpcalc.exe` in a folder on PATH.

## Library folders

The web app works on the videos inside the configured folders, listed in
`framekit.json` (see `framekit.example.json`) or managed with
`python -m framekit roots add|remove PATH`. Folders are set server-side only,
so the browser can never point the app elsewhere on disk. Uploads from the web
app land in `work/uploads`, which is always part of the library.

Everything the suite generates lives under `work/` by default: the index
database, cached fingerprints, extracted frames and audio, exports, colour
match renders, and the quarantine. It is gitignored and safe to delete. Your
video files are only ever touched by an explicit quarantine or "move into
group folders", and both can be undone.

## Sound categories

The sound library starts with a small list of categories: ambience, foley,
footsteps, impacts, voice, music and other. Replace or extend it in
`framekit.json`:

```json
"sounds": {"categories": ["ambience", "foley", "footsteps", "impacts", "voice", "music", "whooshes", "other"]}
```

Each category is a subfolder of the sound folder. Files you copy into the
folder yourself appear after "Rescan folder", with their subfolder as the
category.

## Directories and environment variables

Every directory can be set with an environment variable, or in a `.env` file
next to this README: copy `.env.example` to `.env` and uncomment what you
need. A variable already set in the environment wins over the file, relative
paths are taken from this folder, and lists use `;` between folders on
Windows. `.env` is gitignored.

| Variable | What it sets | Default |
|---|---|---|
| `FRAMEKIT_WORK` | index, caches, thumbnails, extracted frames | `work` |
| `FRAMEKIT_CONFIG` | saved folder list and tool settings | `framekit.json` |
| `FRAMEKIT_UPLOADS` | uploaded videos (always a library folder) | `work\uploads` |
| `FRAMEKIT_EXPORTS` | zips, contact sheets, exported frames and groups | `work\exports` |
| `FRAMEKIT_QUARANTINE` | files set aside by duplicate removal (keep on the same drive as your videos) | `work\quarantine` |
| `FRAMEKIT_HANDOFF_DIR` | frames saved as handoff PNGs from Lineage | `work\lineage\handoff` |
| `FRAMEKIT_SOUNDS` | the sound library (one subfolder per category) | `work\sounds` |
| `FRAMEKIT_COLORMATCH_DIR` | colour match sessions and rendered videos | `work\colormatch` |
| `FRAMEKIT_ROOTS` | library folders, added to the saved ones | |
| `FRAMEKIT_LINEAGE_DIRS` | lineage folders, added to the saved ones (shown as "from .env") | |
| `FRAMEKIT_FRAME_INBOX` | Unique frames drop folder, if none is saved | `work\frames-inbox` |
| `FRAMEKIT_HOST`, `FRAMEKIT_PORT` | web app address | `127.0.0.1`, `8082` |
| `HF_HUB_OFFLINE=1` | never contact Hugging Face once the CLIP model is downloaded | |

`python -m framekit dirs` prints where everything ends up. Library scans skip
FrameKit's own folders even if you place them inside a library folder.

## Unique frames drop folder

Drop images on the Unique frames page, or copy them into `work\frames-inbox`.
To use another folder, set `FRAMEKIT_FRAME_INBOX`, or add `"unique": {"inbox": "D:/Frames"}` to `framekit.json` (which wins).
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
python -m framekit scenes clip.mp4 --list                     # show the cuts; drop --list to write clips
python -m framekit lineage add D:/Generated/finished
python -m framekit lineage scan                              # or: lineage scan D:/SomeFolder (just that one)
python -m framekit lineage tree c007
python -m framekit lineage faces c007                        # likeness of every take to the original face
python -m framekit lineage end c007_g03_t2 58                # end a take at frame 58 ("full" to undo)
python -m framekit lineage export c007_g03_t2 58             # save frame 58 as a handoff PNG
python -m framekit sounds split ambience.wav --category ambience --dry-run   # list, then save, every sound between gaps
python -m framekit colormatch D:/Gen/shot1 --reference start.png  # a folder of segments, in name order
python -m framekit colormatch-image handoff.png start.png          # match one frame to the original
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
- `framekit/tools/` has one module per tool: `frames`, `scenes`, `lineage`, `colormatch`, `audio`, `resize`, `reverse`, `dupes`,
  `grouping`, `unique`, `analysis`, and `editor/`.
- `start-framekit.bat` and `scripts/` hold the Windows launcher and the
  start-at-sign-in installer.
- `framekit/web/` is the Flask app. There is one template per page, with shared
  `static/app.js` and `app.css` and no build step.
- `framekit/cli.py` is the command line.
