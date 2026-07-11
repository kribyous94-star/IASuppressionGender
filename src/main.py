"""IASuppressionGender — orchestrateur (venv core).

Lance chaque détecteur dans son propre venv (sous-processus), fusionne les
détections, puis rend la vidéo avec les frames flaguées remplacées par du noir.
"""
import argparse
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


def parse_args():
    p = argparse.ArgumentParser(
        prog="IASuppressionGender",
        description="Noircit chaque frame d'une vidéo où un individu du genre "
                    "ciblé est présent (audio conservé, 100%% hors ligne).")
    p.add_argument("-i", "--input", required=True, help="vidéo d'entrée")
    p.add_argument("-g", "--gender", required=True,
                   choices=sorted(GENDER_ALIASES),
                   help="genre ciblé (homme/femme ou male/female)")
    p.add_argument("-o", "--output", default=None,
                   help="vidéo de sortie (défaut : <input>_censored.mp4)")
    p.add_argument("--detectors", default="face,body",
                   help="détecteurs, séparés par des virgules (défaut : face,body)")
    p.add_argument("--stride", type=int, default=3,
                   help="analyse 1 frame sur N (défaut : 3)")
    p.add_argument("--pad", type=float, default=0.25,
                   help="marge noircie autour de chaque détection, en secondes")
    p.add_argument("--gap", type=float, default=0.5,
                   help="fusionne deux zones noires séparées de moins de N secondes")
    p.add_argument("--face-thr", type=float, default=None,
                   help="seuil de confiance du détecteur visage (défaut : 0.55)")
    p.add_argument("--body-thr", type=float, default=None,
                   help="seuil de confiance du détecteur corps (défaut : 0.60)")
    p.add_argument("--strict", action="store_true",
                   help="noircit aussi les personnes au genre incertain")
    p.add_argument("--keep-work", action="store_true",
                   help="conserve les JSON d'analyse dans work/ (réutilisés au prochain run)")
    return p.parse_args()


def run_detector(name, video, work_dir, stride):
    """Lance un détecteur dans son venv et renvoie son JSON de résultats."""
    spec = DETECTORS[name]
    out_path = work_dir / f"{name}.json"
    if out_path.exists():
        print(f"[{name}] résultats existants réutilisés ({out_path})")
        with open(out_path) as f:
            return json.load(f)

    py = ROOT / "venvs" / spec["venv"] / "bin" / "python"
    if not py.exists():
        sys.exit(f"venv '{spec['venv']}' manquant : lancez ./install.sh")

    cmd = [str(py), str(ROOT / spec["script"]),
           "--video", str(video),
           "--output", str(out_path),
           "--stride", str(stride),
           "--project-root", str(ROOT)]
    print(f"[{name}] analyse en cours…")
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"[{name}] terminé en {time.time() - t0:.1f}s")
    with open(out_path) as f:
        return json.load(f)


def main():
    args = parse_args()
    target = GENDER_ALIASES[args.gender]

    video = Path(args.input).resolve()
    if not video.is_file():
        sys.exit(f"vidéo introuvable : {video}")
    output = Path(args.output).resolve() if args.output else \
        video.parent / f"{video.stem}_censored.mp4"

    names = [n.strip() for n in args.detectors.split(",") if n.strip()]
    unknown = [n for n in names if n not in DETECTORS]
    if unknown:
        sys.exit(f"détecteur(s) inconnu(s) : {unknown} — disponibles : {list(DETECTORS)}")

    thresholds = {
        "face": args.face_thr if args.face_thr is not None else DETECTORS["face"]["default_thr"],
        "body": args.body_thr if args.body_thr is not None else DETECTORS["body"]["default_thr"],
    }

    work_dir = ROOT / "work" / video.stem
    work_dir.mkdir(parents=True, exist_ok=True)

    # 1. Analyse — un sous-processus par détecteur, chacun dans son venv
    results = {name: run_detector(name, video, work_dir, args.stride)
               for name in names}

    # 2. Fusion des détections → ensemble de frames à noircir
    meta = next(iter(results.values()))
    fps, total = meta["fps"], meta["total_frames"]
    flagged = fuse(results, target=target, thresholds=thresholds,
                   fps=fps, total_frames=total,
                   pad_s=args.pad, gap_s=args.gap, strict=args.strict)
    pct = 100 * len(flagged) / max(total, 1)
    print(f"[fusion] {len(flagged)}/{total} frames noircies ({pct:.1f}%) "
          f"— cible : {target}, strict : {args.strict}")

    # 3. Rendu : frames noires + remux audio
    render(video, output, flagged)
    print(f"[ok] vidéo écrite : {output}")

    if not args.keep_work:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
