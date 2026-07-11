"""IASuppressionGender — interface en ligne de commande (venv core).

Lancée par ./cli.sh ; l'interface graphique (./run.sh → ui.py) utilise le même
moteur (pipeline.py).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import (DETECTORS, GENDER_ALIASES, render_from_ranges,  # noqa: E402
                      run_job)

OUT_CHOICES = {"video": ("video",), "plages": ("ranges",),
               "both": ("video", "ranges")}


def parse_args():
    p = argparse.ArgumentParser(
        prog="IASuppressionGender",
        description="Noircit chaque frame d'une vidéo où un individu du genre "
                    "ciblé est présent (audio conservé, 100%% hors ligne).")
    p.add_argument("-i", "--input", required=True, help="vidéo d'entrée")
    p.add_argument("-g", "--gender",
                   choices=sorted(GENDER_ALIASES),
                   help="genre ciblé (homme/femme ou male/female) — "
                        "requis sauf avec --ranges")
    p.add_argument("-o", "--output", default=None,
                   help="vidéo de sortie (défaut : <input>_censored.mp4)")
    p.add_argument("--out", default="both", choices=sorted(OUT_CHOICES),
                   help="sorties à produire : la vidéo censurée, le fichier "
                        "de plages (JSON éditable), ou les deux (défaut)")
    p.add_argument("--ranges", default=None, metavar="PLAGES.json",
                   help="rend la vidéo directement depuis un fichier de "
                        "plages (éventuellement édité) : aucune détection "
                        "n'est relancée, -g/--out sont ignorés")
    p.add_argument("--ranges-out", default=None,
                   help="chemin du fichier de plages "
                        "(défaut : <sortie>.plages.json)")
    p.add_argument("--detectors", default="face,body",
                   help="détecteurs, séparés par des virgules (défaut : face,body)")
    p.add_argument("--stride", type=int, default=3,
                   help="analyse 1 frame sur N (défaut : 3)")
    p.add_argument("--pad", type=float, default=0.25,
                   help="marge noircie autour de chaque détection, en secondes")
    p.add_argument("--gap", type=float, default=0.5,
                   help="fusionne deux zones noires séparées de moins de N secondes")
    p.add_argument("--face-thr", type=float, default=None,
                   help=f"seuil du détecteur visage (défaut : {DETECTORS['face']['default_thr']})")
    p.add_argument("--body-thr", type=float, default=None,
                   help=f"seuil du détecteur corps (défaut : {DETECTORS['body']['default_thr']})")
    p.add_argument("--strict", action="store_true",
                   help="noircit aussi les personnes au genre incertain")
    p.add_argument("--keep-work", action="store_true",
                   help="conserve les JSON d'analyse dans work/ (réutilisés au prochain run)")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        if args.ranges:
            render_from_ranges(video=args.input, ranges=args.ranges,
                               output=args.output)
            return
        if not args.gender:
            sys.exit("-g/--gender est requis (sauf avec --ranges)")
        thresholds = {}
        if args.face_thr is not None:
            thresholds["face"] = args.face_thr
        if args.body_thr is not None:
            thresholds["body"] = args.body_thr
        run_job(video=args.input, gender=args.gender, output=args.output,
                detectors=[n.strip() for n in args.detectors.split(",") if n.strip()],
                stride=args.stride, pad=args.pad, gap=args.gap,
                thresholds=thresholds, strict=args.strict,
                keep_work=args.keep_work,
                outputs=OUT_CHOICES[args.out], ranges_out=args.ranges_out)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
