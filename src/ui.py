"""IASuppressionGender — interface graphique locale (venv core, lancée par ./run.sh).

Interface web servie sur 127.0.0.1 uniquement (Gradio embarqué, aucun accès
réseau externe). Utilise le même moteur que la CLI (pipeline.py).

Flux en deux étapes :
1. Analyser la vidéo — ou importer un fichier de plages existant pour sauter
   l'analyse — → tableau de plages éditables (désactiver, ajuster en secondes
   ou en h:mm:ss, ajouter des plages) ;
2. Générer les sorties choisies (vidéo censurée et/ou fichier de plages JSON)
   à partir du tableau, sans relancer l'analyse.
"""
import argparse
import asyncio
import json
import queue
import sys
import threading
from pathlib import Path

# Sur Windows, ProactorEventLoop leve ConnectionResetError (WinError 10054)
# a chaque fermeture de connexion navigateur. SelectorEventLoop l'evite.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import gradio as gr
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import (DETECTORS, ROOT, render_from_ranges,  # noqa: E402
                      run_job, video_meta, write_ranges_file)
from timefmt import fmt_hms, parse_time  # noqa: E402

GENDER_CHOICES = {"Femme": "female", "Homme": "male"}
OUT_VIDEO = "Vidéo censurée"
OUT_RANGES = "Fichier de plages (JSON)"
TABLE_HEADERS = ["Début (s)", "Fin (s)", "Début (h:m:s)", "Fin (h:m:s)",
                 "Active"]
LOG_MAX_LINES = 60


_nvml = {"handle": None, "tried": False}


def _nvml_handle():
    """Handle NVML du premier GPU NVIDIA (None si absent/inutilisable)."""
    if not _nvml["tried"]:
        _nvml["tried"] = True
        try:
            import pynvml
            pynvml.nvmlInit()
            _nvml["handle"] = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            _nvml["handle"] = None
    return _nvml["handle"]


def _gauge(label, pct):
    if pct is None:
        return f"<span style='opacity:.6'>{label} —</span>"
    color = "#22a559" if pct < 60 else "#e0a800" if pct < 85 else "#dc3545"
    return (f"<span>{label} <b style='color:{color}'>{pct:.0f} %</b></span>")


def system_stats():
    """Ligne HTML : utilisation CPU / RAM / GPU / VRAM en pourcentage."""
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent
    gpu = vram = None
    handle = _nvml_handle()
    if handle is not None:
        try:
            import pynvml
            gpu = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram = 100.0 * mem.used / mem.total
        except Exception:
            pass
    parts = [_gauge("CPU", cpu), _gauge("RAM", ram),
             _gauge("GPU", gpu), _gauge("VRAM", vram)]
    return ("<div style='display:flex;gap:1.5em;font-family:monospace;"
            "font-size:.9em;padding:2px 0'>" + "".join(parts) + "</div>")


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


def _row(start, end, enabled=True):
    return [round(start, 3), round(end, 3), fmt_hms(start), fmt_hms(end),
            bool(enabled)]


def _row_bounds(row):
    """(start, end) d'une ligne : colonnes secondes, sinon colonnes h:m:s."""
    start = parse_time(row[0] if len(row) > 0 else None)
    end = parse_time(row[1] if len(row) > 1 else None)
    if start is None and len(row) > 2:
        start = parse_time(row[2])
    if end is None and len(row) > 3:
        end = parse_time(row[3])
    return start, end


def _table_to_ranges(table):
    """Tableau de l'interface (liste de lignes) → plages [{start, end, enabled}]."""
    ranges = []
    for row in table or []:
        start, end = _row_bounds(row)
        if start is None or end is None or end <= start:
            continue  # ligne vide ou incomplète
        ranges.append({"start": round(start, 3), "end": round(end, 3),
                       "enabled": bool(row[4]) if len(row) > 4 else True})
    return ranges


def _ranges_to_table(ranges):
    rows = []
    for r in ranges:
        start, end = parse_time(r.get("start")), parse_time(r.get("end"))
        if start is None or end is None:
            continue
        rows.append(_row(start, end, r.get("enabled", True)))
    return rows


def sync_table(prev, table):
    """Après édition d'une cellule, resynchronise secondes ↔ h:mm:ss.

    Compare avec l'état précédent pour savoir quelle colonne l'utilisateur a
    modifiée : les secondes mettent à jour le h:m:s, et inversement.
    """
    table = [list(r) + [True] * (5 - len(r)) for r in (table or [])]
    same_shape = prev and len(prev) == len(table)
    for i, row in enumerate(table):
        p = prev[i] if same_shape else None
        for sec_col, hms_col in ((0, 2), (1, 3)):
            sec, hms = parse_time(row[sec_col]), parse_time(row[hms_col])
            if p is not None and sec == parse_time(p[sec_col]) \
                    and hms is not None and hms != parse_time(p[hms_col]):
                # l'utilisateur a édité la colonne h:m:s → elle fait foi
                row[sec_col] = round(hms, 3)
                row[hms_col] = fmt_hms(hms)
            elif sec is not None:
                row[sec_col] = round(sec, 3)
                row[hms_col] = fmt_hms(sec)
    return table, [list(r) for r in table]


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
            yield item, gr.update(), gr.update(), gr.update()
            continue
        lines = item
        lines.append(f"✅ Analyse terminée : {len(stats['ranges'])} plage(s), "
                     f"{stats['flagged']}/{stats['total']} frames concernées "
                     f"({stats['pct']:.1f}%). Vérifiez/éditez le tableau puis "
                     f"lancez l'étape 2.")
        table = _ranges_to_table(stats["ranges"])
        state = {"video": video, "gender": GENDER_CHOICES[genre],
                 "fps": stats["fps"], "total": stats["total"],
                 "settings": {"detectors": list(detectors),
                              "stride": int(stride), "pad": float(pad),
                              "gap": float(gap), "strict": bool(strict)}}
        yield ("\n".join(lines[-LOG_MAX_LINES:]), table, state,
               [list(r) for r in table])


def import_ranges(video, ranges_file, genre):
    """Alternative à l'étape 1 : charge un fichier de plages existant
    (aucune analyse), pour l'éditer puis générer."""
    if not ranges_file:
        raise gr.Error("Choisissez un fichier de plages (JSON).")
    if not video:
        raise gr.Error("Choisissez aussi la vidéo correspondante (à gauche).")
    try:
        with open(ranges_file) as f:
            data = json.load(f)
        ranges = data["ranges"]
    except (OSError, json.JSONDecodeError, KeyError) as e:
        raise gr.Error(f"fichier de plages illisible : {e}")

    table = _ranges_to_table(ranges)
    fps, total = video_meta(video)
    state = {"video": video,
             "gender": data.get("gender", GENDER_CHOICES[genre]),
             "fps": fps, "total": total,
             "settings": data.get("settings", {})}
    log = (f"✅ {len(table)} plage(s) importée(s) depuis "
           f"{Path(ranges_file).name} (aucune analyse). Vérifiez/éditez le "
           f"tableau puis lancez l'étape 2.")
    return log, table, state, [list(r) for r in table]


def add_range(table):
    """Ajoute une ligne à ajuster (0 s → 1 s, active)."""
    table = (table or []) + [_row(0.0, 1.0)]
    return table, [list(r) for r in table]


def generate(state, table, outs):
    """Étape 2 : produit les sorties choisies à partir du tableau édité."""
    if not state:
        raise gr.Error("Lancez d'abord l'étape 1 (analyse ou import de plages).")
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
            "**Étape 1** : choisissez une vidéo puis analysez-la — ou importez "
            "un fichier de plages existant pour sauter l'analyse. "
            "**Étape 2** : vérifiez/éditez les plages (désactivez, ajustez le "
            "début/la fin en secondes ou en h:mm:ss, ajoutez-en), puis générez "
            "la vidéo censurée et/ou le fichier de plages. "
            "Traitement 100 % local et hors ligne.")

        state = gr.State()
        prev_table = gr.State()
        monitor = gr.HTML(system_stats())
        timer = gr.Timer(2)
        timer.tick(system_stats, outputs=[monitor], show_progress="hidden")
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
                with gr.Accordion("…ou importer des plages existantes "
                                  "(sans analyse)", open=False):
                    ranges_in = gr.File(label="Fichier de plages (JSON)",
                                        file_types=[".json"])
                    import_btn = gr.Button("Importer les plages")
                log = gr.Textbox(label="Journal", lines=12, max_lines=12,
                                 interactive=False)
            with gr.Column():
                table = gr.Dataframe(
                    headers=TABLE_HEADERS,
                    datatype=["number", "number", "str", "str", "bool"],
                    column_count=(5, "fixed"),
                    type="array",
                    interactive=True,
                    label="Plages à noircir — éditez au choix les secondes ou "
                          "le h:mm:ss (synchronisés), décochez pour désactiver")
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
                          outputs=[log, table, state, prev_table])
        import_btn.click(import_ranges,
                         inputs=[video_in, ranges_in, genre],
                         outputs=[log, table, state, prev_table])
        table.input(sync_table, inputs=[prev_table, table],
                    outputs=[table, prev_table])
        add_btn.click(add_range, inputs=[table],
                      outputs=[table, prev_table])
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
