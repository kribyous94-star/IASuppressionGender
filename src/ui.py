"""IASuppressionGender — interface graphique locale (venv core, lancée par ./run.sh).

Interface web servie sur 127.0.0.1 uniquement (Gradio embarqué, aucun accès
réseau externe). Utilise le même moteur que la CLI (pipeline.py).

Flux en deux étapes :
1. Analyser la vidéo — ou importer un fichier de plages existant pour sauter
   l'analyse — → tableau de plages éditables ;
2. Éditer les plages (action : cacher l'image, cacher une zone, couper le son,
   sauter ; début/fin par saisie ou « = position vidéo » ; clic sur une plage
   → la vidéo saute à son début ; aperçu filtré appliqué en direct au lecteur),
   puis générer les sorties choisies sans relancer l'analyse.
"""
import argparse
import json
import queue
import sys
import threading
from pathlib import Path

import gradio as gr
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import (ACTIONS, DEFAULT_ZONE, DETECTORS, ROOT,  # noqa: E402
                      render_from_ranges, run_job, video_meta,
                      write_ranges_file)
from timefmt import fmt_hms, parse_time  # noqa: E402

GENDER_CHOICES = {"Femme": "female", "Homme": "male"}
OUT_VIDEO = "Vidéo censurée"
OUT_RANGES = "Fichier de plages (JSON)"

# Libellés des actions — mêmes intitulés que l'éditeur d'ummah-verse
ACTION_LABELS = {
    "HIDE_VIDEO": "Cacher l'image entière",
    "HIDE_ZONE": "Cacher une zone",
    "MUTE_AUDIO": "Couper le son",
    "SKIP": "Passer (sauter la plage)",
}
LABEL_TO_ACTION = {v: k for k, v in ACTION_LABELS.items()}

TABLE_HEADERS = ["Début (s)", "Fin (s)", "Début (h:m:s)", "Fin (h:m:s)",
                 "Action", "Zone x,y,w,h (%)", "Message", "Active"]
COL_ACTION, COL_ZONE, COL_MSG, COL_ON = 4, 5, 6, 7
ROW_DEFAULTS = [None, None, "", "", ACTION_LABELS["HIDE_VIDEO"], "", "", True]
LOG_MAX_LINES = 60

# Lit la position courante du lecteur (0 si aucune vidéo chargée)
JS_VIDEO_TIME = ("() => { const v = document.querySelector('#iasg-video video');"
                 " return v ? v.currentTime : 0; }")

# Aperçu filtré : applique en direct les plages au lecteur de gauche (écran
# noir + message, zones cachées, son coupé, sauts de lecture), débrayable via
# la case « Aperçu filtré » — même logique que l'éditeur d'ummah-verse.
HEAD_JS = """
<script>
(function () {
  const S = { ranges: [], apply: true };
  window.iasgSetRanges = (json) => {
    try { S.ranges = JSON.parse(json || "[]"); } catch { S.ranges = []; }
    update();
  };
  window.iasgSetApply = (v) => { S.apply = !!v; update(); };

  const video = () => document.querySelector('#iasg-video video');

  function frameBox(v) {  // image réelle dans l'élément (letterbox exclue)
    if (!v.videoWidth || !v.videoHeight) return null;
    const W = v.clientWidth, H = v.clientHeight;
    const s = Math.min(W / v.videoWidth, H / v.videoHeight);
    const w = v.videoWidth * s, h = v.videoHeight * s;
    return { left: v.offsetLeft + (W - w) / 2, top: v.offsetTop + (H - h) / 2,
             w, h };
  }

  function overlay(v) {
    let ov = v.parentElement.querySelector('.iasg-overlay');
    if (!ov) {
      ov = document.createElement('div');
      ov.className = 'iasg-overlay';
      ov.style.cssText = 'position:absolute;inset:0;pointer-events:none;z-index:10;';
      if (getComputedStyle(v.parentElement).position === 'static')
        v.parentElement.style.position = 'relative';
      v.parentElement.appendChild(ov);
    }
    return ov;
  }

  const esc = (s) => { const d = document.createElement('div');
                       d.textContent = s; return d.innerHTML; };

  function update() {
    const v = video();
    if (!v) return;
    const ov = overlay(v), t = v.currentTime;
    let cover = null; const zones = []; let mute = false;
    if (S.apply) {
      for (const r of S.ranges) {
        if (r.enabled === false || t < r.start || t >= r.end) continue;
        if (r.action === 'HIDE_VIDEO' && !cover) cover = r;
        else if (r.action === 'HIDE_ZONE' && r.zone) zones.push(r.zone);
        else if (r.action === 'MUTE_AUDIO') mute = true;
        else if (r.action === 'SKIP' && !v.paused) {
          v.currentTime = r.end + 0.05; return;
        }
      }
    }
    v.muted = mute;
    let html = '';
    if (cover) {
      html = '<div style="position:absolute;inset:0;background:#000;' +
        'display:flex;align-items:center;justify-content:center;color:#ddd;' +
        'text-align:center;padding:1em;">' +
        esc(cover.message || 'Scène Masquée') + '</div>';
    } else {
      const fb = zones.length ? frameBox(v) : null;
      if (fb) for (const z of zones) {
        html += '<div style="position:absolute;background:#000;' +
          `left:${fb.left + z.x / 100 * fb.w}px;top:${fb.top + z.y / 100 * fb.h}px;` +
          `width:${z.w / 100 * fb.w}px;height:${z.h / 100 * fb.h}px;"></div>`;
      }
    }
    ov.innerHTML = html;
  }

  setInterval(() => {  // (ré)attache le handler quand le lecteur (ré)apparaît
    const v = video();
    if (v && !v.dataset.iasg) {
      v.dataset.iasg = '1';
      for (const ev of ['timeupdate', 'seeked', 'pause', 'loadedmetadata'])
        v.addEventListener(ev, update);
    }
  }, 800);
})();
</script>
"""


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
    return (f"<span>{label} <b style='color:{color}'>{pct:.0f} %</b></span>")


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


# ── Conversions tableau ↔ plages ─────────────────────────────────────────────

def _action_of(cell):
    """Cellule « Action » → action du format (libellé français ou brut)."""
    s = str(cell or "").strip()
    return LABEL_TO_ACTION.get(s, s if s in ACTIONS else "HIDE_VIDEO")


def _zone_to_str(zone):
    if not zone:
        return ""
    return (f"{zone['x']:g},{zone['y']:g},{zone['w']:g},{zone['h']:g}")


def _zone_from_str(s):
    parts = str(s or "").replace(";", ",").split(",")
    try:
        x, y, w, h = (float(p.replace(",", ".").strip()) for p in parts)
    except ValueError:
        return None
    return {"x": x, "y": y, "w": w, "h": h}


def _row(start, end, action="HIDE_VIDEO", zone=None, message="", enabled=True):
    return [round(start, 3), round(end, 3), fmt_hms(start), fmt_hms(end),
            ACTION_LABELS.get(action, ACTION_LABELS["HIDE_VIDEO"]),
            _zone_to_str(zone), str(message or ""), bool(enabled)]


def _pad_row(row):
    row = list(row)[:len(ROW_DEFAULTS)]
    return row + ROW_DEFAULTS[len(row):]


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
    """Tableau de l'interface → plages [{start, end, action, …, enabled}]."""
    ranges = []
    for row in table or []:
        row = _pad_row(row)
        start, end = _row_bounds(row)
        if start is None or end is None or end <= start:
            continue  # ligne vide ou incomplète
        action = _action_of(row[COL_ACTION])
        r = {"start": round(start, 3), "end": round(end, 3),
             "action": action, "enabled": bool(row[COL_ON])}
        if action == "HIDE_ZONE":
            r["zone"] = _zone_from_str(row[COL_ZONE]) or dict(DEFAULT_ZONE)
        if action == "HIDE_VIDEO" and str(row[COL_MSG]).strip():
            r["message"] = str(row[COL_MSG]).strip()
        ranges.append(r)
    return ranges


def _ranges_to_table(ranges):
    rows = []
    for r in ranges:
        start, end = parse_time(r.get("start")), parse_time(r.get("end"))
        if start is None or end is None:
            continue
        action = r.get("action", "HIDE_VIDEO")
        if action not in ACTIONS:
            continue
        rows.append(_row(start, end, action, r.get("zone"),
                         r.get("message", ""), r.get("enabled", True)))
    return rows


def _ranges_json(table):
    """Plages du tableau → JSON poussé au lecteur (aperçu filtré en direct)."""
    return json.dumps(_table_to_ranges(table))


def sync_table(prev, table):
    """Après édition d'une cellule, resynchronise secondes ↔ h:mm:ss.

    Compare avec l'état précédent pour savoir quelle colonne l'utilisateur a
    modifiée : les secondes mettent à jour le h:m:s, et inversement.
    """
    table = [_pad_row(r) for r in (table or [])]
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
    return table, [list(r) for r in table], _ranges_json(table)


# ── Étape 1 : analyse ou import ──────────────────────────────────────────────

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
            yield item, gr.update(), gr.update(), gr.update(), gr.update()
            continue
        lines = item
        lines.append(f"✅ Analyse terminée : {len(stats['ranges'])} plage(s), "
                     f"{stats['flagged']}/{stats['total']} frames concernées "
                     f"({stats['pct']:.1f}%). Vérifiez/éditez le tableau puis "
                     f"lancez l'étape 2.")
        table = _ranges_to_table(stats["ranges"])
        state = {"video": video, "gender": GENDER_CHOICES[genre],
                 "fps": stats["fps"], "total": stats["total"]}
        yield ("\n".join(lines[-LOG_MAX_LINES:]), table, state,
               [list(r) for r in table], _ranges_json(table))


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
    state = {"video": video, "gender": GENDER_CHOICES[genre],
             "fps": fps, "total": total}
    ignored = len(ranges) - len(table)
    log = (f"✅ {len(table)} plage(s) importée(s) depuis "
           f"{Path(ranges_file).name} (aucune analyse)"
           + (f" — {ignored} plage(s) invalide(s) ignorée(s)"
              if ignored else "")
           + ". Vérifiez/éditez le tableau puis lancez l'étape 2.")
    title = data.get("title") or gr.update()
    return (log, table, state, [list(r) for r in table],
            _ranges_json(table), title)


# ── Étape 2 : édition des plages ─────────────────────────────────────────────

def _panel_for(row):
    """Contenu du panneau d'édition pour une ligne du tableau."""
    start, end = _row_bounds(row)
    action = _action_of(row[COL_ACTION])
    zone = _zone_from_str(row[COL_ZONE]) or dict(DEFAULT_ZONE)
    md = (f"**Plage sélectionnée** : {fmt_hms(start or 0)} → "
          f"{fmt_hms(end or 0)}")
    return (md, gr.update(value=ACTION_LABELS[action]),
            gr.update(visible=action == "HIDE_ZONE"),
            zone["x"], zone["y"], zone["w"], zone["h"],
            gr.update(value=str(row[COL_MSG] or ""),
                      visible=action == "HIDE_VIDEO"))


_PANEL_NOOP = (gr.update(),) * 8


def on_select(table, evt: gr.SelectData):
    """Clic sur une ligne : sélectionne la plage, remplit le panneau
    d'édition et fait sauter la vidéo au début de la plage."""
    row_idx = evt.index[0] if evt.index else None
    table = [_pad_row(r) for r in (table or [])]
    if row_idx is None or not (0 <= row_idx < len(table)):
        return (None, "Aucune plage sélectionnée.", *_PANEL_NOOP[1:],
                gr.update(), gr.update(visible=False))
    start, _ = _row_bounds(table[row_idx])
    seek = start if start is not None else gr.update()
    return (row_idx, *_panel_for(table[row_idx]), seek,
            gr.update(visible=True))


def _apply_to_row(sel, table, fn):
    """Applique fn(row) à la ligne sélectionnée et resynchronise tout."""
    table = [_pad_row(r) for r in (table or [])]
    if sel is None or not (0 <= sel < len(table)):
        return (gr.update(), gr.update(), gr.update(), gr.update())
    fn(table[sel])
    row = table[sel]
    start, end = _row_bounds(row)
    if start is not None:
        row[0], row[2] = round(start, 3), fmt_hms(start)
    if end is not None:
        row[1], row[3] = round(end, 3), fmt_hms(end)
    md = (f"**Plage sélectionnée** : {fmt_hms(start or 0)} → "
          f"{fmt_hms(end or 0)}")
    return table, [list(r) for r in table], _ranges_json(table), md


def set_action(sel, label, table):
    action = LABEL_TO_ACTION.get(label, "HIDE_VIDEO")

    def fn(row):
        row[COL_ACTION] = ACTION_LABELS[action]
        if action == "HIDE_ZONE" and not _zone_from_str(row[COL_ZONE]):
            row[COL_ZONE] = _zone_to_str(DEFAULT_ZONE)

    out = _apply_to_row(sel, table, fn)
    zone = _zone_from_str(_pad_row(table[sel])[COL_ZONE]) \
        if sel is not None and 0 <= sel < len(table or []) else None
    zone = zone or dict(DEFAULT_ZONE)
    return (*out, gr.update(visible=action == "HIDE_ZONE"),
            zone["x"], zone["y"], zone["w"], zone["h"],
            gr.update(visible=action == "HIDE_VIDEO"))


def set_zone(sel, x, y, w, h, table):
    def fn(row):
        row[COL_ZONE] = _zone_to_str(
            {"x": x or 0, "y": y or 0, "w": w or 0, "h": h or 0})
    return _apply_to_row(sel, table, fn)


def set_message(sel, msg, table):
    def fn(row):
        row[COL_MSG] = msg or ""
    return _apply_to_row(sel, table, fn)


def set_start(sel, pos, table):
    def fn(row):
        row[0] = round(float(pos or 0), 3)
        row[2] = fmt_hms(row[0])
    return _apply_to_row(sel, table, fn)


def set_end(sel, pos, table):
    def fn(row):
        row[1] = round(float(pos or 0), 3)
        row[3] = fmt_hms(row[1])
    return _apply_to_row(sel, table, fn)


def delete_range(sel, table):
    table = [_pad_row(r) for r in (table or [])]
    if sel is not None and 0 <= sel < len(table):
        del table[sel]
    return (table, [list(r) for r in table], _ranges_json(table),
            None, gr.update(visible=False))


def add_range(pos, table):
    """Ajoute une plage de 5 s à la position courante de la vidéo."""
    t = round(float(pos or 0), 3)
    table = [_pad_row(r) for r in (table or [])] + [_row(t, t + 5.0)]
    return table, [list(r) for r in table], _ranges_json(table)


# ── Étape 2 : génération ─────────────────────────────────────────────────────

def generate(state, table, outs, title):
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
        ranges_file = write_ranges_file(out_dir / f"{stem}.plages.json",
                                        ranges, title=title.strip() or "Hide")

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
            "**Étape 2** : cliquez sur une plage pour l'éditer (la vidéo saute "
            "à son début) : action (cacher l'image, cacher une zone, couper le "
            "son, sauter), début/fin (saisie ou « = position vidéo »), aperçu "
            "filtré en direct sur le lecteur. Puis générez la vidéo censurée "
            "et/ou le fichier de plages (format partagé « ummahverse-filter-"
            "list »). Traitement 100 % local et hors ligne.")

        state = gr.State()
        prev_table = gr.State()
        sel_row = gr.State()
        cur_pos = gr.Number(visible=False)   # position vidéo lue côté client
        seek_pos = gr.Number(visible=False)  # position à atteindre (clic plage)
        ranges_json = gr.Textbox(visible=False)  # plages → aperçu filtré JS
        monitor = gr.HTML(system_stats())
        timer = gr.Timer(2)
        timer.tick(system_stats, outputs=[monitor], show_progress="hidden")

        with gr.Row():
            with gr.Column():
                video_in = gr.Video(label="Vidéo d'entrée", sources=["upload"],
                                    elem_id="iasg-video")
                apply_chk = gr.Checkbox(
                    value=True, label="Aperçu filtré",
                    info="applique les plages au lecteur ci-dessus en direct "
                         "(décochez pour voir la vidéo brute)")
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
                    datatype=["number", "number", "str", "str", "str", "str",
                              "str", "bool"],
                    column_count=(len(TABLE_HEADERS), "fixed"),
                    type="array",
                    interactive=True,
                    label="Plages — cliquez sur une ligne pour l'éditer "
                          "ci-dessous (la vidéo saute à son début), décochez "
                          "« Active » pour désactiver")
                add_btn = gr.Button("➕ Ajouter une plage à la position vidéo")
                with gr.Group(visible=False) as panel:
                    sel_md = gr.Markdown("Aucune plage sélectionnée.")
                    with gr.Row():
                        action_dd = gr.Dropdown(
                            list(ACTION_LABELS.values()),
                            label="Action", scale=2)
                        start_pos_btn = gr.Button("⏱ Début = position vidéo")
                        end_pos_btn = gr.Button("⏱ Fin = position vidéo")
                        del_btn = gr.Button("🗑 Supprimer", variant="stop")
                    with gr.Group(visible=False) as zone_grp:
                        with gr.Row():
                            zx = gr.Number(label="Zone x (%)", minimum=0,
                                           maximum=100)
                            zy = gr.Number(label="Zone y (%)", minimum=0,
                                           maximum=100)
                            zw = gr.Number(label="Largeur (%)", minimum=0,
                                           maximum=100)
                            zh = gr.Number(label="Hauteur (%)", minimum=0,
                                           maximum=100)
                        gr.Markdown(
                            "*Zone en % de l'image (x,y = coin haut-gauche). "
                            "Visible en direct sur l'aperçu filtré.*")
                    msg_tb = gr.Textbox(
                        label="Message affiché sur l'écran noir",
                        placeholder="Scène Masquée", visible=False)
                title_tb = gr.Textbox(label="Titre de la liste", value="Hide")
                outs = gr.CheckboxGroup(
                    [OUT_VIDEO, OUT_RANGES], value=[OUT_VIDEO, OUT_RANGES],
                    label="Sorties à générer")
                generate_btn = gr.Button("2️⃣ Générer", variant="primary")
                video_out = gr.Video(label="Vidéo censurée", interactive=False)
                ranges_out = gr.File(label="Fichier de plages (JSON)",
                                     interactive=False)

        row_outs = [table, prev_table, ranges_json]
        panel_outs = [sel_md, action_dd, zone_grp, zx, zy, zw, zh, msg_tb]

        analyse_btn.click(analyse,
                          inputs=[video_in, genre, detectors, stride, pad, gap,
                                  face_thr, body_thr, strict, keep_work],
                          outputs=[log, table, state, prev_table, ranges_json])
        import_btn.click(import_ranges,
                         inputs=[video_in, ranges_in, genre],
                         outputs=[log, table, state, prev_table, ranges_json,
                                  title_tb])
        table.input(sync_table, inputs=[prev_table, table],
                    outputs=row_outs)
        table.select(on_select, inputs=[table],
                     outputs=[sel_row, *panel_outs, seek_pos, panel])

        # panneau d'édition de la plage sélectionnée
        action_dd.input(set_action, inputs=[sel_row, action_dd, table],
                        outputs=[*row_outs, sel_md, zone_grp,
                                 zx, zy, zw, zh, msg_tb])
        for z in (zx, zy, zw, zh):
            z.input(set_zone, inputs=[sel_row, zx, zy, zw, zh, table],
                    outputs=[*row_outs, sel_md])
        msg_tb.input(set_message, inputs=[sel_row, msg_tb, table],
                     outputs=[*row_outs, sel_md])
        start_pos_btn.click(None, inputs=None, outputs=[cur_pos],
                            js=JS_VIDEO_TIME) \
            .then(set_start, inputs=[sel_row, cur_pos, table],
                  outputs=[*row_outs, sel_md])
        end_pos_btn.click(None, inputs=None, outputs=[cur_pos],
                          js=JS_VIDEO_TIME) \
            .then(set_end, inputs=[sel_row, cur_pos, table],
                  outputs=[*row_outs, sel_md])
        del_btn.click(delete_range, inputs=[sel_row, table],
                      outputs=[*row_outs, sel_row, panel])
        add_btn.click(None, inputs=None, outputs=[cur_pos],
                      js=JS_VIDEO_TIME) \
            .then(add_range, inputs=[cur_pos, table], outputs=row_outs)

        # liaisons avec le lecteur (aperçu filtré + navigation)
        ranges_json.change(None, inputs=[ranges_json], outputs=None,
                           js="(s) => { window.iasgSetRanges(s); }")
        apply_chk.change(None, inputs=[apply_chk], outputs=None,
                         js="(v) => { window.iasgSetApply(v); }")
        seek_pos.change(None, inputs=[seek_pos], outputs=None,
                        js="(t) => { const v = document.querySelector("
                           "'#iasg-video video');"
                           " if (v && t != null && isFinite(t))"
                           " v.currentTime = Math.max(0, t); }")

        generate_btn.click(generate,
                           inputs=[state, table, outs, title_tb],
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
                               share=False, head=HEAD_JS)


if __name__ == "__main__":
    main()
