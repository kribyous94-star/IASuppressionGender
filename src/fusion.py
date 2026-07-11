"""Fusion des détections : OR entre détecteurs, seuils, expansion temporelle.

Biais assumé vers le faux positif : mieux vaut une frame noire en trop qu'un
individu du genre ciblé visible une frame.
"""

OTHER = {"male": "female", "female": "male"}


def _hit(det, target, thr, strict):
    """Une détection individuelle déclenche-t-elle le noircissement ?"""
    gender, conf = det.get("gender", "unknown"), float(det.get("conf", 0.0))
    if gender == target and conf >= thr:
        return True
    if strict:
        # personne présente dont on ne peut PAS exclure le genre cible :
        # genre inconnu, ou classée dans l'autre genre mais sous le seuil
        if gender == "unknown":
            return True
        if gender == OTHER[target] and conf < thr:
            return True
    return False


def fuse(results, target, thresholds, fps, total_frames,
         pad_s=0.25, gap_s=0.5, strict=False):
    """Combine les JSON des détecteurs → set d'indices de frames à noircir.

    results     : {nom: json_du_détecteur} (cf. ARCHITECTURE.md §4.1)
    thresholds  : {nom: seuil de confiance}
    """
    flagged = set()
    for name, res in results.items():
        thr = thresholds.get(name, 0.5)
        stride = int(res.get("stride", 1))
        for idx_str, dets in res.get("frames", {}).items():
            if any(_hit(d, target, thr, strict) for d in dets):
                # la frame analysée i couvre [i, i+stride)
                i = int(idx_str)
                flagged.update(range(i, min(i + stride, total_frames)))

    if not flagged:
        return flagged

    # spans contigus → pad → fusion des gaps courts
    spans = _to_spans(sorted(flagged))
    pad = round(pad_s * fps)
    spans = [(max(0, a - pad), min(total_frames - 1, b + pad)) for a, b in spans]
    spans = _merge_gaps(spans, round(gap_s * fps))

    out = set()
    for a, b in spans:
        out.update(range(a, b + 1))
    return out


def _to_spans(sorted_indices):
    spans, start, prev = [], sorted_indices[0], sorted_indices[0]
    for i in sorted_indices[1:]:
        if i != prev + 1:
            spans.append((start, prev))
            start = i
        prev = i
    spans.append((start, prev))
    return spans


def _merge_gaps(spans, max_gap):
    merged = [spans[0]]
    for a, b in spans[1:]:
        pa, pb = merged[-1]
        if a - pb <= max_gap:
            merged[-1] = (pa, max(pb, b))
        else:
            merged.append((a, b))
    return merged
