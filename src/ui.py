"""IASuppressionGender — interface graphique locale (venv core, lancée par ./run.sh).

Interface web servie sur 127.0.0.1 uniquement (Gradio embarqué, aucun accès
réseau externe). Utilise le même moteur que la CLI (pipeline.py).

Flux en deux étapes :
1. Analyser la vidéo — ou importer un fichier de plages existant pour sauter
   l'analyse — → liste de plages éditables ;
2. Éditer les plages, présentées en cartes façon éditeur ummah-verse : chaque
   plage a ses boutons ▶ (aller à la plage), ⏱ (= position vidéo, début et
   fin), son action (cacher l'image, cacher une zone, couper le son, sauter),
   sa case Active et sa suppression ; sélection multiple + suppression en
   masse ; aperçu filtré appliqué en direct au lecteur. Puis générer les
   sorties choisies sans relancer l'analyse.
"""
import argparse
import json
import math
import queue
import sys
import threading
import uuid
from functools import partial
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

PAGE_SIZE = 20  # plages par page, comme l'éditeur d'ummah-verse
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
  window.iasgSeek = (t) => {
    const v = video();
    if (v && isFinite(t)) v.currentTime = Math.max(0, t);
  };

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


# ── Modèle : liste de plages (état) ──────────────────────────────────────────
# Une plage = {id, start, end, action, zone|None, message, enabled}

def _new_id():
    return uuid.uuid4().hex[:12]


def _new_range(start, end, action="HIDE_VIDEO"):
    return {"id": _new_id(), "start": round(start, 3), "end": round(end, 3),
            "action": action, "zone": None, "message": "", "enabled": True}


def _norm_range(r):
    """Plage venant d'un fichier → plage normalisée (None si invalide)."""
    start, end = parse_time(r.get("start")), parse_time(r.get("end"))
    action = r.get("action", "HIDE_VIDEO")
    if start is None or end is None or end <= start or action not in ACTIONS:
        return None
    zone = r.get("zone")
    return {"id": str(r.get("id") or _new_id())[:40],
            "start": round(start, 3), "end": round(end, 3), "action": action,
            "zone": dict(zone) if isinstance(zone, dict) else None,
            "message": str(r.get("message") or ""),
            "enabled": bool(r.get("enabled", True))}


def _ranges_json(ranges):
    """Plages → JSON poussé au lecteur (aperçu filtré en direct)."""
    return json.dumps(ranges or [])


def _fmt_s(v):
    return f"{v:g}"


def _patch(ranges, rid, **patch):
    return [({**r, **patch} if r["id"] == rid else r) for r in ranges or []]


def _clamp_page(page, ranges):
    pages = max(1, math.ceil(len(ranges or []) / PAGE_SIZE))
    return min(max(int(page or 1), 1), pages)


# ── Mutations d'une plage (rid lié à la carte au rendu) ──────────────────────

def set_bound(rid, which, text, ranges):
    """Commit d'un champ temps (secondes ou h:mm:ss) ; invalide → inchangé."""
    v = parse_time(text)
    if v is not None and v >= 0:
        ranges = _patch(ranges, rid, **{which: round(v, 3)})
    return ranges, _ranges_json(ranges)


def set_bound_pos(rid, which, pos, ranges):
    """Bouton ⏱ : début/fin = position courante du lecteur."""
    ranges = _patch(ranges, rid, **{which: round(float(pos or 0), 3)})
    return ranges, _ranges_json(ranges)


def set_action(rid, label, ranges):
    action = LABEL_TO_ACTION.get(label, "HIDE_VIDEO")
    cur = next((r for r in ranges or [] if r["id"] == rid), None)
    zone = (cur or {}).get("zone")
    if action == "HIDE_ZONE" and not zone:
        zone = dict(DEFAULT_ZONE)
    ranges = _patch(ranges, rid, action=action, zone=zone)
    return ranges, _ranges_json(ranges)


def set_zone(rid, x, y, w, h, ranges):
    zone = {"x": float(x or 0), "y": float(y or 0),
            "w": float(w or 0), "h": float(h or 0)}
    ranges = _patch(ranges, rid, zone=zone)
    return ranges, _ranges_json(ranges)


def set_message(rid, msg, ranges):
    ranges = _patch(ranges, rid, message=str(msg or ""))
    return ranges, _ranges_json(ranges)


def set_enabled(rid, val, ranges):
    ranges = _patch(ranges, rid, enabled=bool(val))
    return ranges, _ranges_json(ranges)


def toggle_sel(rid, val, selected):
    sel = set(selected or [])
    (sel.add if val else sel.discard)(rid)
    return sorted(sel)


def delete_range(rid, ranges, selected, page):
    ranges = [r for r in ranges or [] if r["id"] != rid]
    selected = [i for i in selected or [] if i != rid]
    return (ranges, selected, _ranges_json(ranges),
            _clamp_page(page, ranges))


def delete_selected(ranges, selected, page):
    sel = set(selected or [])
    ranges = [r for r in ranges or [] if r["id"] not in sel]
    return ranges, [], _ranges_json(ranges), _clamp_page(page, ranges)


def toggle_page_sel(page_ids, all_selected, selected):
    sel = set(selected or [])
    sel = sel - set(page_ids) if all_selected else sel | set(page_ids)
    return sorted(sel)


def add_range(pos, ranges):
    """➕ : nouvelle plage de 5 s à la position courante du lecteur."""
    t = round(float(pos or 0), 3)
    ranges = list(ranges or []) + [_new_range(t, t + 5.0)]
    return (ranges, _ranges_json(ranges),
            _clamp_page(len(ranges), ranges))  # → dernière page


def page_delta(delta, page, ranges):
    return _clamp_page((page or 1) + delta, ranges)


# ── Étape 1 : analyse ou import ──────────────────────────────────────────────

def analyse(video, genre, detectors, stride, pad, gap,
            face_thr, body_thr, strict, keep_work):
    """Étape 1 : détection + fusion → liste de plages éditables."""
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
            yield (item,) + (gr.update(),) * 5
            continue
        lines = item
        lines.append(f"✅ Analyse terminée : {len(stats['ranges'])} plage(s), "
                     f"{stats['flagged']}/{stats['total']} frames concernées "
                     f"({stats['pct']:.1f}%). Vérifiez/éditez les plages puis "
                     f"lancez l'étape 2.")
        ranges = [_new_range(r["start"], r["end"]) for r in stats["ranges"]]
        state = {"video": video, "gender": GENDER_CHOICES[genre],
                 "fps": stats["fps"], "total": stats["total"]}
        yield ("\n".join(lines[-LOG_MAX_LINES:]), ranges, state,
               _ranges_json(ranges), [], 1)


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
        raw = data["ranges"]
    except (OSError, json.JSONDecodeError, KeyError) as e:
        raise gr.Error(f"fichier de plages illisible : {e}")

    ranges = [r for r in (_norm_range(r) for r in raw) if r]
    fps, total = video_meta(video)
    state = {"video": video, "gender": GENDER_CHOICES[genre],
             "fps": fps, "total": total}
    ignored = len(raw) - len(ranges)
    log = (f"✅ {len(ranges)} plage(s) importée(s) depuis "
           f"{Path(ranges_file).name} (aucune analyse)"
           + (f" — {ignored} plage(s) invalide(s) ignorée(s)"
              if ignored else "")
           + ". Vérifiez/éditez les plages puis lancez l'étape 2.")
    title = data.get("title") or gr.update()
    return log, ranges, state, _ranges_json(ranges), [], 1, title


# ── Étape 2 : génération ─────────────────────────────────────────────────────

def generate(state, ranges, outs, title):
    """Étape 2 : produit les sorties choisies à partir des plages éditées."""
    if not state:
        raise gr.Error("Lancez d'abord l'étape 1 (analyse ou import de plages).")
    if not outs:
        raise gr.Error("Choisissez au moins une sortie (vidéo ou fichier).")
    ranges = ranges or []

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


def ranges_editor(ranges, selected, page, C):
    """Corps du @gr.render : liste des plages en cartes façon ummah-verse.

    C : composants persistants de build_app (états, JSON de l'aperçu,
    position vidéo). Chaque carte porte ses propres commandes : sélection,
    ▶ aller à la plage, début/fin (champ + bouton ⏱ « = position vidéo »),
    action et champs associés (message / zone), case Active, suppression.
    """
    ranges = ranges or []
    sel = set(selected or [])
    page = _clamp_page(page, ranges)
    pages = max(1, math.ceil(len(ranges) / PAGE_SIZE))
    shown = ranges[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]
    page_ids = [r["id"] for r in shown]
    all_sel = bool(page_ids) and all(i in sel for i in page_ids)
    ranges_state, sel_state = C["ranges_state"], C["sel_state"]
    page_state, ranges_json = C["page_state"], C["ranges_json"]
    cur_pos = C["cur_pos"]

    # barre d'outils : compteur, sélection de page, suppression en masse
    with gr.Row():
        gr.Markdown(f"**{len(ranges)} plage(s)**"
                    + (f" — {len(sel)} sélectionnée(s)" if sel else ""))
        if page_ids:
            psel_btn = gr.Button(
                "☐ Désélectionner la page" if all_sel
                else "☑ Sélectionner la page",
                size="sm", scale=0, min_width=190)
            psel_btn.click(partial(toggle_page_sel, page_ids, all_sel),
                           inputs=[sel_state], outputs=[sel_state])
        if sel:
            dsel_btn = gr.Button(f"🗑 Supprimer la sélection ({len(sel)})",
                                 size="sm", scale=0, variant="stop",
                                 min_width=200)
            dsel_btn.click(delete_selected,
                           inputs=[ranges_state, sel_state, page_state],
                           outputs=[ranges_state, sel_state, ranges_json,
                                    page_state])

    if not ranges:
        gr.Markdown("*Aucune plage : lancez l'analyse, importez un fichier, "
                    "ou ➕ ajoutez une plage à la main.*")

    for offset, r in enumerate(shown):
        rid = r["id"]
        n = (page - 1) * PAGE_SIZE + offset + 1
        with gr.Group():
            with gr.Row():
                sel_chk = gr.Checkbox(value=rid in sel, show_label=False,
                                      container=False, scale=0, min_width=28)
                goto_btn = gr.Button("▶", size="sm", scale=0, min_width=36)
                gr.Markdown(f"**{n}.** {_fmt_s(r['start'])} s → "
                            f"{_fmt_s(r['end'])} s")
                on_chk = gr.Checkbox(value=r["enabled"], label="Active",
                                     container=False, scale=0, min_width=90)
                del_btn = gr.Button("🗑", size="sm", scale=0, min_width=36,
                                    variant="stop")
            with gr.Row():
                start_tb = gr.Textbox(value=fmt_hms(r["start"]),
                                      label="Début (s ou h:m:s)", scale=2,
                                      min_width=110)
                start_pos_btn = gr.Button("⏱", size="sm", scale=0,
                                          min_width=36)
                end_tb = gr.Textbox(value=fmt_hms(r["end"]),
                                    label="Fin (s ou h:m:s)", scale=2,
                                    min_width=110)
                end_pos_btn = gr.Button("⏱", size="sm", scale=0, min_width=36)
                action_dd = gr.Dropdown(list(ACTION_LABELS.values()),
                                        value=ACTION_LABELS[r["action"]],
                                        label="Action", scale=3, min_width=170)
            if r["action"] == "HIDE_VIDEO":
                msg_tb = gr.Textbox(value=r["message"],
                                    label="Message affiché sur l'écran noir",
                                    placeholder="Scène Masquée")
                for ev in (msg_tb.submit, msg_tb.blur):
                    ev(partial(set_message, rid),
                       inputs=[msg_tb, ranges_state],
                       outputs=[ranges_state, ranges_json])
            elif r["action"] == "HIDE_ZONE":
                zone = r["zone"] or dict(DEFAULT_ZONE)
                with gr.Row():
                    zs = [gr.Number(value=zone[k], label=lbl, minimum=0,
                                    maximum=100, min_width=80)
                          for k, lbl in (("x", "Zone x (%)"), ("y", "y (%)"),
                                         ("w", "Largeur (%)"),
                                         ("h", "Hauteur (%)"))]
                for z in zs:
                    for ev in (z.submit, z.blur):
                        ev(partial(set_zone, rid),
                           inputs=[*zs, ranges_state],
                           outputs=[ranges_state, ranges_json])

            sel_chk.input(partial(toggle_sel, rid),
                          inputs=[sel_chk, sel_state], outputs=[sel_state])
            on_chk.input(partial(set_enabled, rid),
                         inputs=[on_chk, ranges_state],
                         outputs=[ranges_state, ranges_json])
            goto_btn.click(  # aller à la plage (côté client, sans requête)
                None, inputs=None, outputs=None,
                js=f"() => window.iasgSeek({r['start']})")
            for ev in (start_tb.submit, start_tb.blur):
                ev(partial(set_bound, rid, "start"),
                   inputs=[start_tb, ranges_state],
                   outputs=[ranges_state, ranges_json])
            for ev in (end_tb.submit, end_tb.blur):
                ev(partial(set_bound, rid, "end"),
                   inputs=[end_tb, ranges_state],
                   outputs=[ranges_state, ranges_json])
            start_pos_btn.click(None, inputs=None, outputs=[cur_pos],
                                js=JS_VIDEO_TIME) \
                .then(partial(set_bound_pos, rid, "start"),
                      inputs=[cur_pos, ranges_state],
                      outputs=[ranges_state, ranges_json])
            end_pos_btn.click(None, inputs=None, outputs=[cur_pos],
                              js=JS_VIDEO_TIME) \
                .then(partial(set_bound_pos, rid, "end"),
                      inputs=[cur_pos, ranges_state],
                      outputs=[ranges_state, ranges_json])
            action_dd.input(partial(set_action, rid),
                            inputs=[action_dd, ranges_state],
                            outputs=[ranges_state, ranges_json])
            del_btn.click(partial(delete_range, rid),
                          inputs=[ranges_state, sel_state, page_state],
                          outputs=[ranges_state, sel_state, ranges_json,
                                   page_state])

    if pages > 1:
        with gr.Row():
            prev_btn = gr.Button("◀ Page précédente", size="sm",
                                 interactive=page > 1)
            gr.Markdown(f"Page **{page} / {pages}**")
            next_btn = gr.Button("Page suivante ▶", size="sm",
                                 interactive=page < pages)
            prev_btn.click(partial(page_delta, -1),
                           inputs=[page_state, ranges_state],
                           outputs=[page_state])
            next_btn.click(partial(page_delta, 1),
                           inputs=[page_state, ranges_state],
                           outputs=[page_state])


def build_app():
    with gr.Blocks(title="IASuppressionGender") as app:
        gr.Markdown(
            "# IASuppressionGender\n"
            "**Étape 1** : choisissez une vidéo puis analysez-la — ou importez "
            "un fichier de plages existant pour sauter l'analyse. "
            "**Étape 2** : éditez les plages — ▶ pour aller à la plage dans "
            "la vidéo, ⏱ pour caler un début/une fin sur la position du "
            "lecteur, action au choix (cacher l'image, cacher une zone, "
            "couper le son, sauter), sélection multiple pour supprimer en "
            "masse, aperçu filtré en direct — puis générez la vidéo censurée "
            "et/ou le fichier de plages (format partagé « ummahverse-filter-"
            "list »). Traitement 100 % local et hors ligne.")

        state = gr.State()
        ranges_state = gr.State([])
        sel_state = gr.State([])
        page_state = gr.State(1)
        cur_pos = gr.Number(visible=False)   # position vidéo lue côté client
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
                add_btn = gr.Button("➕ Ajouter une plage à la position vidéo")

                comps = {"ranges_state": ranges_state, "sel_state": sel_state,
                         "page_state": page_state, "ranges_json": ranges_json,
                         "cur_pos": cur_pos}

                @gr.render(inputs=[ranges_state, sel_state, page_state])
                def _render(ranges, selected, page):
                    ranges_editor(ranges, selected, page, comps)

                title_tb = gr.Textbox(label="Titre de la liste", value="Hide")
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
                          outputs=[log, ranges_state, state, ranges_json,
                                   sel_state, page_state])
        import_btn.click(import_ranges,
                         inputs=[video_in, ranges_in, genre],
                         outputs=[log, ranges_state, state, ranges_json,
                                  sel_state, page_state, title_tb])
        add_btn.click(None, inputs=None, outputs=[cur_pos],
                      js=JS_VIDEO_TIME) \
            .then(add_range, inputs=[cur_pos, ranges_state],
                  outputs=[ranges_state, ranges_json, page_state])

        # liaisons avec le lecteur (aperçu filtré en direct)
        ranges_json.change(None, inputs=[ranges_json], outputs=None,
                           js="(s) => { window.iasgSetRanges(s); }")
        apply_chk.change(None, inputs=[apply_chk], outputs=None,
                         js="(v) => { window.iasgSetApply(v); }")

        generate_btn.click(generate,
                           inputs=[state, ranges_state, outs, title_tb],
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
