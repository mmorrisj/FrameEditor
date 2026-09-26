"""The tools of the suite. Each is independent and exposes plain functions:

frames    - break a video into image frames (all, every Nth, fixed rate, scene changes, keyframes)
audio     - pull the audio track out untouched, or convert it
dupes     - find duplicate videos (exact, re-encoded, trimmed) and quarantine extras
grouping  - cluster videos by the similarity of sampled frames
analysis  - shots, duplicate frames, clusters and ordering within one frame set
editor    - fix regions of individual frames and repackage with the original audio
"""
