"""Moteur du pipeline (venv core) — partagé par la CLI (main.py) et
l'interface (ui.py).

Lance chaque détecteur dans son propre venv (sous-processus), fusionne les
détections, puis rend la vidéo avec les frames flaguées remplacées par du noir.
"""
import datetime
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT / "src"))
from fusion import (fuse, render_plan, spans_to_frames,  # noqa: E402
                    spans_to_seconds)
from render import render  # noqa: E402
from timefmt import parse_time  # noqa: E402

# Actions du format « ummahverse-filter-list » (cf. ARCHITECTURE.md §4.5)
ACTIONS = ("HIDE_VIDEO", "HIDE_ZONE", "MUTE_AUDIO", "SKIP")
DEFAULT_ZONE = {"x": 25.0, "y": 25.0, "w": 50.0, "h": 50.0}

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


def video_meta(video):
    """(fps, nombre de frames) d'une vidéo."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"impossible d'ouvrir {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total


def _range_id():
    """Identifiant de plage façon JS (`Date.now().toString(36)` + aléa)."""
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    n, s = int(time.time() * 1000), ""
    while n:
        n, r = divmod(n, 36)
        s = alphabet[r] + s
    return s + "".join(random.choices(alphabet, k=2))


def _seconds(value):
    """Secondes arrondies au millième, en entier si la valeur est ronde."""
    v = round(parse_time(value) or 0.0, 3)
    return int(v) if v == int(v) else v


def _clean_zone(zone):
    """Zone {x,y,w,h} en % de l'image, bornée à [0,100] (None si invalide)."""
    try:
        x = min(100.0, max(0.0, float(zone["x"])))
        y = min(100.0, max(0.0, float(zone["y"])))
        w = min(100.0 - x, max(0.1, float(zone["w"])))
        h = min(100.0 - y, max(0.1, float(zone["h"])))
    except (TypeError, KeyError, ValueError):
        return None
    return {"x": round(x, 1), "y": round(y, 1),
            "w": round(w, 1), "h": round(h, 1)}


def write_ranges_file(path, ranges, title="Hide"):
    """Écrit le fichier de plages au format « ummahverse-filter-list »
    (cf. ARCHITECTURE.md §4.5), partagé avec d'autres logiciels.

    start/end (secondes) font foi ; à la lecture ils acceptent aussi une
    chaîne « h:mm:ss.mmm ». Champs par action (mêmes règles que la
    validation d'ummah-verse) : message pour HIDE_VIDEO (défaut « Scène
    Masquée »), zone {x,y,w,h en %} pour HIDE_ZONE. La clé `enabled`
    (propre à ce logiciel) n'est écrite que pour les plages désactivées,
    afin que les fichiers restent conformes au format commun.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    for r in sorted(ranges, key=lambda r: parse_time(r.get("start")) or 0.0):
        action = r.get("action", "HIDE_VIDEO")
        if action not in ACTIONS:
            action = "HIDE_VIDEO"
        entry = {
            "id": r.get("id") or _range_id(),
            "end": _seconds(r.get("end")),
            "start": _seconds(r.get("start")),
            "action": action,
        }
        if action == "HIDE_VIDEO":
            entry["message"] = str(r.get("message") or "Scène Masquée")
        elif action == "HIDE_ZONE":
            entry["zone"] = _clean_zone(r.get("zone")) or dict(DEFAULT_ZONE)
        if r.get("label"):
            entry["label"] = str(r["label"])
        if not r.get("enabled", True):
            entry["enabled"] = False
        entries.append(entry)
    data = {
        "format": "ummahverse-filter-list",
        "version": 1,
        "title": title,
        "exportedAt": datetime.datetime.now(datetime.timezone.utc)
                      .isoformat(timespec="milliseconds")
                      .replace("+00:00", "Z"),
        "ranges": entries,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return str(path)


def render_from_ranges(video, ranges, output=None, log=print):
    """Rend la vidéo censurée à partir de plages en secondes — soit la liste
    [{start, end, action, …}], soit le chemin d'un fichier de plages (JSON),
    éventuellement édité à la main. Toutes les actions du format sont
    appliquées (image cachée, zone cachée, son coupé, plage sautée).
    Aucune détection n'est relancée.
    """
    video = Path(video).resolve()
    if not video.is_file():
        raise FileNotFoundError(f"vidéo introuvable : {video}")
    if isinstance(ranges, (str, Path)):
        with open(ranges) as f:
            ranges = json.load(f)["ranges"]
    output = Path(output).resolve() if output else \
        video.parent / f"{video.stem}_censored.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)

    fps, total = video_meta(video)
    active = [r for r in ranges if r.get("enabled", True)]
    plan = render_plan(ranges, fps, total)
    flagged = plan["black"]
    pct = 100 * len(flagged) / max(total, 1)
    log(f"[rendu] {len(active)} plage(s) active(s) : {len(flagged)}/{total} "
        f"frames noircies ({pct:.1f}%), {len(plan['zones'])} avec zone cachée, "
        f"{len(plan['skip'])} sautées, {len(plan['mute_s'])} plage(s) sans son")
    render(video, output, plan,
           progress=lambda i, t: log(f"[rendu] {i}/{t} frames"))
    log(f"[ok] vidéo écrite : {output}")
    return {"output": str(output), "flagged": len(flagged),
            "total": total, "pct": pct}


def run_job(video, gender, output=None, detectors=("face", "body"), stride=3,
            pad=0.25, gap=0.5, thresholds=None, strict=False, keep_work=False,
            log=print, capture=False, outputs=("video", "ranges"),
            ranges_out=None):
    """Traite une vidéo de bout en bout. Renvoie un dict de statistiques
    (dont la liste des plages en secondes).

    gender  : 'homme'/'femme'/'male'/'female' — thresholds : {détecteur: seuil}.
    outputs : sorties à produire, parmi 'video' et 'ranges' (fichier de
              plages JSON). L'analyse a lieu dans tous les cas et les plages
              sont toujours renvoyées dans les stats.
    """
    target = GENDER_ALIASES[str(gender).lower()]
    video = Path(video).resolve()
    if not video.is_file():
        raise FileNotFoundError(f"vidéo introuvable : {video}")
    output = Path(output).resolve() if output else \
        video.parent / f"{video.stem}_censored.mp4"
    ranges_path = Path(ranges_out).resolve() if ranges_out else \
        output.parent / f"{output.stem}.plages.json"

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

        # 2. Fusion des détections → plages de frames à noircir
        meta = next(iter(results.values()))
        fps, total = meta["fps"], meta["total_frames"]
        spans = fuse(results, target=target, thresholds=thr,
                     fps=fps, total_frames=total,
                     pad_s=pad, gap_s=gap, strict=strict)
        ranges = spans_to_seconds(spans, fps)
        flagged = spans_to_frames(spans)
        pct = 100 * len(flagged) / max(total, 1)
        log(f"[fusion] {len(ranges)} plage(s), {len(flagged)}/{total} frames "
            f"noircies ({pct:.1f}%) — cible : {target}, strict : {strict}")

        stats = {"output": None, "ranges_file": None, "ranges": ranges,
                 "flagged": len(flagged), "total": total, "pct": pct,
                 "fps": fps}

        # 3. Sorties demandées : fichier de plages et/ou vidéo
        if "ranges" in outputs:
            stats["ranges_file"] = write_ranges_file(ranges_path, ranges)
            log(f"[ok] plages écrites : {ranges_path}")
        if "video" in outputs:
            output.parent.mkdir(parents=True, exist_ok=True)
            log("[rendu] écriture de la vidéo…")
            render(video, output, {"black": flagged})
            stats["output"] = str(output)
            log(f"[ok] vidéo écrite : {output}")
    finally:
        if not keep_work:
            shutil.rmtree(work_dir, ignore_errors=True)

    return stats
