"""Moteur du pipeline (venv core) — partagé par la CLI (main.py) et
l'interface (ui.py).

Lance chaque détecteur dans son propre venv (sous-processus), fusionne les
détections, puis rend la vidéo avec les frames flaguées remplacées par du noir.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT / "src"))
from fusion import fuse  # noqa: E402
from render import render  # noqa: E402

# Registre des détecteurs : en ajouter un = ajouter une entrée ici
# (+ son venv dans install.sh et son script dans src/detectors/).
DETECTORS = {
    "face": {
        "venv": "face",
        "script": "src/detectors/face_detector.py",
        "default_thr": 0.55,
    },
    "body": {
        "venv": "body",
        "script": "src/detectors/body_detector.py",
        "default_thr": 0.60,
    },
}

GENDER_ALIASES = {
    "homme": "male", "male": "male", "m": "male",
    "femme": "female", "female": "female", "f": "female",
}


def detector_env(venv_name, plain_progress=False):
    """Environnement du sous-processus : libs CUDA pip du venv (si installées)
    exposées via LD_LIBRARY_PATH, car rien n'est installé au niveau système."""
    env = os.environ.copy()
    venv_dir = ROOT / "venvs" / venv_name
    nvlibs = sorted(str(p) for p in venv_dir.glob(
        "lib/python*/site-packages/nvidia/*/lib"))
    if nvlibs:
        prev = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(nvlibs + ([prev] if prev else []))
    if plain_progress:
        env["IASG_PROGRESS"] = "plain"
    return env


def run_detector(name, video, work_dir, stride, log=print, capture=False):
    """Lance un détecteur dans son venv et renvoie son JSON de résultats.

    capture=True : la sortie du sous-processus est relue ligne par ligne et
    transmise à `log` (mode interface) au lieu d'hériter du terminal.
    """
    spec = DETECTORS[name]
    out_path = work_dir / f"{name}.json"
    if out_path.exists():
        log(f"[{name}] résultats existants réutilisés")
        with open(out_path) as f:
            return json.load(f)

    py = ROOT / "venvs" / spec["venv"] / "bin" / "python"
    if not py.exists():
        raise RuntimeError(f"venv '{spec['venv']}' manquant : lancez ./install.sh")

    cmd = [str(py), str(ROOT / spec["script"]),
           "--video", str(video),
           "--output", str(out_path),
           "--stride", str(stride),
           "--project-root", str(ROOT)]
    log(f"[{name}] analyse en cours…")
    t0 = time.time()
    env = detector_env(spec["venv"], plain_progress=capture)
    if capture:
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            line = line.strip()
            if line:
                log(f"[{name}] {line}")
        if proc.wait() != 0:
            raise RuntimeError(f"le détecteur '{name}' a échoué (voir journal)")
    else:
        subprocess.run(cmd, check=True, env=env)
    log(f"[{name}] terminé en {time.time() - t0:.1f}s")
    with open(out_path) as f:
        return json.load(f)


def run_job(video, gender, output=None, detectors=("face", "body"), stride=3,
            pad=0.25, gap=0.5, thresholds=None, strict=False, keep_work=False,
            log=print, capture=False):
    """Traite une vidéo de bout en bout. Renvoie un dict de statistiques.

    gender : 'homme'/'femme'/'male'/'female' — thresholds : {détecteur: seuil}.
    """
    target = GENDER_ALIASES[str(gender).lower()]
    video = Path(video).resolve()
    if not video.is_file():
        raise FileNotFoundError(f"vidéo introuvable : {video}")
    output = Path(output).resolve() if output else \
        video.parent / f"{video.stem}_censored.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)

    unknown = [n for n in detectors if n not in DETECTORS]
    if unknown:
        raise ValueError(f"détecteur(s) inconnu(s) : {unknown} — "
                         f"disponibles : {list(DETECTORS)}")
    if not detectors:
        raise ValueError("aucun détecteur sélectionné")

    thr = {n: DETECTORS[n]["default_thr"] for n in DETECTORS}
    thr.update(thresholds or {})

    work_dir = ROOT / "work" / video.stem
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Analyse — un sous-processus par détecteur, chacun dans son venv
        results = {name: run_detector(name, video, work_dir, stride,
                                      log=log, capture=capture)
                   for name in detectors}

        # 2. Fusion des détections → ensemble de frames à noircir
        meta = next(iter(results.values()))
        fps, total = meta["fps"], meta["total_frames"]
        flagged = fuse(results, target=target, thresholds=thr,
                       fps=fps, total_frames=total,
                       pad_s=pad, gap_s=gap, strict=strict)
        pct = 100 * len(flagged) / max(total, 1)
        log(f"[fusion] {len(flagged)}/{total} frames noircies ({pct:.1f}%) "
            f"— cible : {target}, strict : {strict}")

        # 3. Rendu : frames noires + remux audio
        log("[rendu] écriture de la vidéo…")
        render(video, output, flagged)
        log(f"[ok] vidéo écrite : {output}")
    finally:
        if not keep_work:
            shutil.rmtree(work_dir, ignore_errors=True)

    return {"output": str(output), "flagged": len(flagged),
            "total": total, "pct": pct}
