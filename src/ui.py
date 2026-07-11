"""IASuppressionGender — interface graphique locale (venv core, lancée par ./run.sh).

Interface web servie sur 127.0.0.1 uniquement (Gradio embarqué, aucun accès
réseau externe). Utilise le même moteur que la CLI (pipeline.py).
"""
import argparse
import queue
import sys
import threading
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import DETECTORS, ROOT, run_job  # noqa: E402

GENDER_CHOICES = {"Femme": "female", "Homme": "male"}
LOG_MAX_LINES = 60


def process(video, genre, detectors, stride, pad, gap,
            face_thr, body_thr, strict, keep_work):
    """Générateur Gradio : diffuse le journal en direct, puis la vidéo finale."""
    if not video:
        raise gr.Error("Choisissez d'abord une vidéo.")
    if not detectors:
        raise gr.Error("Sélectionnez au moins un détecteur.")

    out = ROOT / "output" / f"{Path(video).stem}_censored.mp4"
    q, result = queue.Queue(), {}

    def worker():
        try:
            result["stats"] = run_job(
                video=video, gender=GENDER_CHOICES[genre], output=out,
                detectors=list(detectors), stride=int(stride),
                pad=float(pad), gap=float(gap),
                thresholds={"face": float(face_thr), "body": float(body_thr)},
                strict=bool(strict), keep_work=bool(keep_work),
                log=q.put, capture=True)
        except Exception as e:  # remonté à l'utilisateur via gr.Error
            result["error"] = e
        finally:
            q.put(None)

    threading.Thread(target=worker, daemon=True).start()

    lines = []
    while True:
        msg = q.get()
        if msg is None:
            break
        lines.append(str(msg))
        yield "\n".join(lines[-LOG_MAX_LINES:]), None

    if "error" in result:
        raise gr.Error(str(result["error"]))
    stats = result["stats"]
    lines.append(f"✅ Terminé : {stats['flagged']}/{stats['total']} frames "
                 f"noircies ({stats['pct']:.1f}%)")
    yield "\n".join(lines[-LOG_MAX_LINES:]), stats["output"]


def build_app():
    with gr.Blocks(title="IASuppressionGender") as app:
        gr.Markdown(
            "# IASuppressionGender\n"
            "Choisissez une vidéo et le genre à supprimer : chaque frame où un "
            "individu de ce genre apparaît sera remplacée par un écran noir "
            "(audio conservé). Traitement 100 % local et hors ligne.")

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
                                    info="deux zones noires plus proches que ça sont fusionnées")
                    face_thr = gr.Slider(0, 1, value=DETECTORS["face"]["default_thr"],
                                         step=0.05, label="Seuil visage")
                    body_thr = gr.Slider(0, 1, value=DETECTORS["body"]["default_thr"],
                                         step=0.05, label="Seuil corps")
                    strict = gr.Checkbox(
                        label="Mode strict",
                        info="noircit aussi les personnes dont le genre est incertain")
                    keep_work = gr.Checkbox(
                        label="Conserver l'analyse (work/)",
                        info="accélère un nouveau passage sur la même vidéo "
                             "avec d'autres réglages")
                launch = gr.Button("Lancer le traitement", variant="primary")
            with gr.Column():
                log = gr.Textbox(label="Journal", lines=18, max_lines=18,
                                 interactive=False)
                video_out = gr.Video(label="Vidéo censurée",
                                     interactive=False)

        launch.click(process,
                     inputs=[video_in, genre, detectors, stride, pad, gap,
                             face_thr, body_thr, strict, keep_work],
                     outputs=[log, video_out])
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
