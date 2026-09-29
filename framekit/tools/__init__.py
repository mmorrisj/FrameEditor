"""The tools of the suite. Each is independent and exposes plain functions:

frames    - break a video into image frames (all, every Nth, fixed rate, scene changes, keyframes)
scenes    - split a video into one clip per scene, with tunable and hand-editable cuts
lineage   - rebuild parent/child chains of AI clips from first and last frames
colormatch - undo color drift across chained AI segments and join them
audio     - pull the audio track out untouched, or convert it
resize    - scale without stretching: keep the shape, or pad / blur / crop to a new one
reverse   - make a rewound copy (optionally faster, or a forwards-then-back boomerang)
dupes     - find duplicate videos (exact, re-encoded, trimmed) and quarantine extras
grouping  - cluster videos by the similarity of sampled frames
analysis  - shots, duplicate frames, clusters and ordering within one frame set
editor    - fix regions of individual frames and repackage with the original audio
"""
