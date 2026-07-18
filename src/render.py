"""Rendu final : réécrit la vidéo en appliquant le plan d'actions (frames
noires, zones cachées, plages sautées), puis remet l'audio original — coupé sur
les plages MUTE_AUDIO/SKIP — via le ffmpeg statique embarqué (imageio-ffmpeg,
installé dans le venv core donc dans le dossier du projet).
"""
import subprocess
import tempfile
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from tqdm import tqdm


def render(video_path, output_path, plan, progress=None):
    """Réécrit la vidéo selon `plan` (cf. fusion.render_plan) :
    black (frames noircies), zones (rectangles noirs par frame), skip
    (frames supprimées) ; skip_s/mute_s sont appliqués à l'audio au remux.

    progress : callback optionnel (frames_traitées, total) appelé périodiquement.
    """
    black_frames = plan.get("black", set())
    zones = plan.get("zones", {})
    skip_frames = plan.get("skip", set())

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
            if idx not in skip_frames:
                if idx in black_frames:
                    frame = black
                else:
                    for z in zones.get(idx, ()):
                        # zone en % de l'image → rectangle noir plein
                        x1 = int(z["x"] / 100 * w)
                        y1 = int(z["y"] / 100 * h)
                        x2 = int((z["x"] + z["w"]) / 100 * w)
                        y2 = int((z["y"] + z["h"]) / 100 * h)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
                writer.write(frame)
            idx += 1
            bar.update(1)
            if progress and idx % step == 0:
                progress(idx, total)
    cap.release()
    writer.release()

    _mux_audio(tmp_path, video_path, output_path,
               mute_s=plan.get("mute_s", ()), skip_s=plan.get("skip_s", ()))
    tmp_path.unlink(missing_ok=True)


def _has_audio(ffmpeg, path):
    """La vidéo a-t-elle une piste audio ? (parse la sortie de ffmpeg -i,
    imageio-ffmpeg n'embarque pas ffprobe)."""
    proc = subprocess.run([ffmpeg, "-hide_banner", "-i", str(path)],
                         capture_output=True, text=True)
    return "Audio:" in proc.stderr


def _mux_audio(rendered, original, output, mute_s=(), skip_s=()):
    """Ré-encode la vidéo rendue en H.264 et y remet l'audio original.

    mute_s : plages (secondes) où l'audio est mis à zéro (MUTE_AUDIO) ;
    skip_s : plages supprimées de la vidéo — l'audio y est découpé aussi
             (aselect + asetpts) pour rester synchrone. Sans plage audio,
             la piste est copiée telle quelle.
    """
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    filters = [f"volume=enable='between(t,{a},{b})':volume=0"
               for a, b in mute_s]
    if skip_s:
        expr = "+".join(f"between(t,{a},{b})" for a, b in skip_s)
        filters += [f"aselect='not({expr})'", "asetpts=N/SR/TB"]
    if filters and not _has_audio(ffmpeg, original):
        filters = []

    audio_args = ["-af", ",".join(filters), "-c:a", "aac"] if filters \
        else ["-c:a", "copy"]
    cmd = [ffmpeg, "-y", "-loglevel", "error",
           "-i", str(rendered), "-i", str(original),
           "-map", "0:v:0", "-map", "1:a?",
           "-c:v", "libx264", "-preset", "fast", "-crf", "20",
           "-pix_fmt", "yuv420p",
           *audio_args, "-shortest",
           str(output)]
    subprocess.run(cmd, check=True)
