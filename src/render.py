"""Rendu final : réécrit la vidéo avec des frames noires, puis remuxe l'audio
original via le ffmpeg statique embarqué (imageio-ffmpeg, installé dans le venv
core donc dans le dossier du projet).
"""
import subprocess
import tempfile
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from tqdm import tqdm


def render(video_path, output_path, flagged_frames, progress=None):
    """Copie la vidéo en noircissant les frames de `flagged_frames` (set d'index).

    progress : callback optionnel (frames_traitées, total) appelé périodiquement.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"impossible d'ouvrir {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False,
                                      dir=Path(output_path).parent)
    tmp_path = Path(tmp.name)
    tmp.close()

    writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))
    black = np.zeros((h, w, 3), dtype=np.uint8)

    idx = 0
    step = max(1, total // 10)
    with tqdm(total=total, desc="[rendu]", unit="f") as bar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(black if idx in flagged_frames else frame)
            idx += 1
            bar.update(1)
            if progress and idx % step == 0:
                progress(idx, total)
    cap.release()
    writer.release()

    _mux_audio(tmp_path, video_path, output_path)
    tmp_path.unlink(missing_ok=True)


def _mux_audio(rendered, original, output):
    """Ré-encode la vidéo rendue en H.264 et y remet l'audio original."""
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg, "-y", "-loglevel", "error",
           "-i", str(rendered), "-i", str(original),
           "-map", "0:v:0", "-map", "1:a?",
           "-c:v", "libx264", "-preset", "fast", "-crf", "20",
           "-pix_fmt", "yuv420p",
           "-c:a", "copy", "-shortest",
           str(output)]
    subprocess.run(cmd, check=True)
