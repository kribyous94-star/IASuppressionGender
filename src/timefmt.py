"""Conversions temps : secondes ↔ « h:mm:ss.mmm » (venv core).

Utilisé par le tableau de l'interface, le fichier de plages et la fusion,
pour que début/fin soient lisibles et éditables dans les deux formats.
"""


def fmt_hms(seconds):
    """90.5 → '0:01:30.500'"""
    seconds = float(seconds)
    h = int(seconds // 3600)
    m = int(seconds % 3600 // 60)
    s = seconds - 3600 * h - 60 * m
    return f"{h}:{m:02d}:{s:06.3f}"


def parse_time(value):
    """Nombre, '90.5', '1:30' ou '0:01:30.5' → secondes (None si invalide)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        parts = [p.strip() for p in str(value).strip().split(":")]
        if not parts or len(parts) > 3 or any(p == "" for p in parts):
            return None
        seconds = 0.0
        for p in parts:
            seconds = seconds * 60 + float(p.replace(",", "."))
        return seconds
    except ValueError:
        return None
