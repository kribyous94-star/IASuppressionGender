"""IASuppressionGender — interface graphique locale (venv core, lancée par ./run.sh).

Interface web servie sur 127.0.0.1 uniquement (Gradio embarqué, aucun accès
réseau externe). Utilise le même moteur que la CLI (pipeline.py).

Flux en deux étapes :
1. Analyser la vidéo → tableau de plages éditables (désactiver, ajuster,
   ajouter des plages) ;
2. Générer les sorties choisies (vidéo censurée et/ou fichier de plages JSON)
   à partir du tableau, sans relancer l'analyse.
"""
import argparse
import queue
import sys
import threading
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import (DETECTORS, ROOT, render_from_ranges,  # noqa: E402
                      run_job, write_ranges_file)

GENDER_CHOICES = {"Femme": "female", "Homme": "male"}
OUT_VIDEO = "Vidéo censurée"
OUT_RANGES = "Fichier de plages (JSON)"
TABLE_HEADERS = ["Début (s)", "Fin (s)", "Active"]
LOG_MAX_LINES = 60


def _stream(worker_fn):
    """Exécute worker_fn(log) dans un thread et diffuse ses lignes de journal.

    Générateur : yield le texte du journal à chaque ligne, puis le résultat.
    """
    q, result = queue.Queue(), {}

    def runner():
        try:
            result["value"] = worker_fn(q.put)
        except Exception as e:  # remonté à l'utilisateur via gr.Error
            result["error"] = e
        finally:
            q.put(None)

    threading.Thread(target=runner, daemon=True).start()
    lines = []
    while True:
        msg = q.get()
        if msg is None:
            break
        lines.append(str(msg))
        yield "\n".join(lines[-LOG_MAX_LINES:]), None
    if "error" in result:
        raise gr.Error(str(result["error"]))
    yield lines, result["value"]


def _table_to_ranges(table):
    """Tableau de l'interface (liste de lignes) → plages [{start, end, enabled}]."""
    ranges = []
    for row in table or []:
        try:
            start, end = float(row[0]), float(row[1])
        except (TypeError, ValueError, IndexError):
            continue  # ligne vide ou incomplète
        if end <= start:
            continue
        ranges.append({"start": round(start, 3), "end": round(end, 3),
                       "enabled": bool(row[2]) if len(row) > 2 else True})
    return ranges


def analyse(video, genre, detectors, stride, pad, gap,
            face_thr, body_thr, strict, keep_work):
    """Étape 1 : détection + fusion → tableau de plages éditables."""
    if not video:
        raise gr.Error("Choisissez d'abord une vidéo.")
    if not detectors:
        raise gr.Error("Sélectionnez au moins un détecteur.")

    def worker(log):
        return run_job(
            video=video, gender=GENDER_CHOICES[genre],
            detectors=list(detectors), stride=int(stride),
            pad=float(pad), gap=float(gap),
            thresholds={"face": float(face_thr), "body": float(body_thr)},
            strict=bool(strict), keep_work=bool(keep_work),
            log=log, capture=True, outputs=())  # les sorties viennent à l'étape 2

    for item, stats in _stream(worker):
        if stats is None:
            yield item, gr.update(), gr.update()
            continue
        lines = item
        lines.append(f"✅ Analyse terminée : {len(stats['ranges'])} plage(s), "
                     f"{stats['flagged']}/{stats['total']} frames concernées "
                     f"({stats['pct']:.1f}%). Vérifiez/éditez le tableau puis "
                     f"lancez l'étape 2.")
        table = [[r["start"], r["end"], r["enabled"]] for r in stats["ranges"]]
        state = {"video": video, "gender": GENDER_CHOICES[genre],
                 "fps": stats["fps"], "total": stats["total"],
                 "settings": {"detectors": list(detectors),
                              "stride": int(stride), "pad": float(pad),
                              "gap": float(gap), "strict": bool(strict)}}
        yield "\n".join(lines[-LOG_MAX_LINES:]), table, state


def add_range(table):
    """Ajoute une ligne vide à ajuster (0 s → 1 s, active)."""
    return (table or []) + [[0.0, 1.0, True]]


def generate(state, table, outs):
    """Étape 2 : produit les sorties choisies à partir du tableau édité."""
    if not state:
        raise gr.Error("Lancez d'abord l'étape 1 (analyse).")
    if not outs:
        raise gr.Error("Choisissez au moins une sortie (vidéo ou fichier).")
    ranges = _table_to_ranges(table)

    stem = Path(state["video"]).stem
    out_dir = ROOT / "output"
    ranges_file = None
    if OUT_RANGES in outs:
        ranges_file = write_ranges_file(
            out_dir / f"{stem}.plages.json", state["video"], state["gender"],
            state["fps"], state["total"], ranges, state["settings"])

    if OUT_VIDEO not in outs:
        done = f"✅ Fichier de plages écrit : {ranges_file}"
        yield done, None, ranges_file
        return

    def worker(log):
        return render_from_ranges(state["video"], ranges,
                                  out_dir / f"{stem}_censored.mp4", log=log)

    for item, stats in _stream(worker):
        if stats is None:
            yield item, None, None
            continue
        lines = item
        if ranges_file:
            lines.append(f"✅ Fichier de plages écrit : {ranges_file}")
        lines.append("✅ Terminé.")
        yield "\n".join(lines[-LOG_MAX_LINES:]), stats["output"], ranges_file


def build_app():
    with gr.Blocks(title="IASuppressionGender") as app:
        gr.Markdown(
            "# IASuppressionGender\n"
            "**Étape 1** : choisissez une vidéo et le genre à supprimer, puis "
            "analysez. **Étape 2** : vérifiez/éditez les plages détectées "
            "(désactivez une plage, ajustez son début/sa fin, ajoutez-en), "
            "puis générez la vidéo censurée et/ou le fichier de plages. "
            "Traitement 100 % local et hors ligne.")

        state = gr.State()
        with gr.Row():
            with gr.Column():
                video_in = gr.Video(label="Vidéo d'entrée", sources=["upload"])
                genre = gr.Radio(list(GENDER_CHOICES), value="Femme",
                                 label="Genre à supprimer")
                with gr.Accordion("Réglages avancés", open=False):
                    detectors = gr.CheckboxGroup(
                        list(DETECTORS), value=list(DETECTORS),
                        label="Détecteurs",
                        info="face : par le visage — body : par le corps/la "
                             "silhouette (fonctionne de dos, visage caché)")
                    stride = gr.Slider(1, 10, value=3, step=1,
                                       label="Stride (analyse 1 frame sur N)",
                                       info="1 = plus précis mais plus lent")
                    pad = gr.Slider(0, 2, value=0.25, step=0.05,
                                    label="Marge de sécurité (s)",
                                    info="durée noircie en plus autour de chaque détection")
                    gap = gr.Slider(0, 3, value=0.5, step=0.1,
                                    label="Fusion des trous (s)",
                                    info="deux plages plus proches que ça sont fusionnées")
                    face_thr = gr.Slider(0, 1, value=DETECTORS["face"]["default_thr"],
                                         step=0.05, label="Seuil visage")
                    body_thr = gr.Slider(0, 1, value=DETECTORS["body"]["default_thr"],
                                         step=0.05, label="Seuil corps")
                    strict = gr.Checkbox(
                        label="Mode strict",
                        info="noircit aussi les personnes dont le genre est incertain")
                    keep_work = gr.Checkbox(
                        label="Conserver l'analyse (work/)",
                        info="accélère une nouvelle analyse de la même vidéo "
                             "avec d'autres réglages")
                analyse_btn = gr.Button("1️⃣ Analyser la vidéo",
                                        variant="primary")
                log = gr.Textbox(label="Journal", lines=12, max_lines=12,
                                 interactive=False)
            with gr.Column():
                table = gr.Dataframe(
                    headers=TABLE_HEADERS,
                    datatype=["number", "number", "bool"],
                    col_count=(3, "fixed"),
                    type="array",
                    interactive=True,
                    label="Plages à noircir (éditables : début/fin en "
                          "secondes, décochez pour désactiver)")
                add_btn = gr.Button("➕ Ajouter une plage")
                outs = gr.CheckboxGroup(
                    [OUT_VIDEO, OUT_RANGES], value=[OUT_VIDEO, OUT_RANGES],
                    label="Sorties à générer")
                generate_btn = gr.Button("2️⃣ Générer", variant="primary")
                video_out = gr.Video(label="Vidéo censurée", interactive=False)
                ranges_out = gr.File(label="Fichier de plages (JSON)",
                                     interactive=False)

        analyse_btn.click(analyse,
                          inputs=[video_in, genre, detectors, stride, pad, gap,
                                  face_thr, body_thr, strict, keep_work],
                          outputs=[log, table, state])
        add_btn.click(add_range, inputs=[table], outputs=[table])
        generate_btn.click(generate,
                           inputs=[state, table, outs],
                           outputs=[log, video_out, ranges_out])
    return app


def main():
    p = argparse.ArgumentParser(description="Interface IASuppressionGender")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--no-browser", action="store_true",
                   help="ne pas ouvrir automatiquement le navigateur")
    args = p.parse_args()

    build_app().queue().launch(server_name="127.0.0.1",
                               server_port=args.port,
                               inbrowser=not args.no_browser,
                               share=False)


if __name__ == "__main__":
    main()
