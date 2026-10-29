"""Cartes de pronostic en image (format vertical, pour réseaux sociaux).

Design original : pas de logos ni d'écussons de clubs (droits des marques).
Chaque match montre le score (ou « VS »), et trois étiquettes tirées du modèle :
  - buts : « +1,5 BUT » ou « -1,5 BUT » ;
  - résultat : 1, N ou 2 ;
  - les deux équipes marquent : « GG » (oui) ou « NG » (non).
Couleurs : bleu = à venir, vert = validé, rouge = raté (après le match).
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dixon_coles import DixonColes

W = 1080
BG = (15, 23, 42)
CARD = (30, 41, 59)
TXT = (241, 245, 249)
MUTED = (148, 163, 184)
BLUE = (59, 130, 246)
GREEN = (34, 197, 94)
RED = (239, 68, 68)
PALETTE = [(99, 102, 241), (14, 165, 233), (16, 185, 129), (245, 158, 11),
           (236, 72, 153), (139, 92, 246), (20, 184, 166), (249, 115, 22)]

_BOLD_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "DejaVuSans-Bold.ttf",
]


def get_font(size):
    for path in _BOLD_FONTS:
        try:
            return ImageFont.truetype(path, size), 0
        except OSError:
            continue
    try:  # police intégrée à Pillow ≥ 10.1 : on simule le gras avec un contour
        return ImageFont.load_default(size=size), 2
    except TypeError:
        return ImageFont.load_default(), 0


def _text(draw, xy, text, size, fill, anchor="mm"):
    font, stroke = get_font(size)
    draw.text(xy, text, font=font, fill=fill, anchor=anchor,
              stroke_width=stroke, stroke_fill=fill)


def _width(draw, text, size):
    font, stroke = get_font(size)
    return draw.textlength(text, font=font) + 2 * stroke


def _fit_lines(draw, text, max_w, start, minimum=28):
    """Réduit la police, puis coupe en deux lignes si le nom est trop long."""
    size = start
    while size > minimum and _width(draw, text, size) > max_w:
        size -= 2
    if _width(draw, text, size) <= max_w or " " not in text:
        return [text], size
    words = text.split()
    best = min(range(1, len(words)),
               key=lambda k: abs(len(" ".join(words[:k])) - len(" ".join(words[k:]))))
    lines = [" ".join(words[:best]), " ".join(words[best:])]
    size = start
    while size > minimum and max(_width(draw, ln, size) for ln in lines) > max_w:
        size -= 2
    return lines, size


# ----------------------------------------------------------------------
# Calculs du modèle
# ----------------------------------------------------------------------
def build_match(model, home, away, league, date_str, hg=None, ag=None):
    """Étiquettes et probabilités d'un match, + validation si le score est connu."""
    m = model.score_matrix(home, away)
    n = m.shape[0]
    total_goals = np.add.outer(np.arange(n), np.arange(n))
    p_over15 = float(m[total_goals >= 2].sum())
    p_btts = float(1 - m[0, :].sum() - m[:, 0].sum() + m[0, 0])
    p1x2 = DixonColes.probs_1x2(m)
    pick = int(np.argmax(p1x2))
    i, j = np.unravel_index(np.argmax(m), m.shape)

    played = hg is not None and ag is not None
    tags = []

    over = p_over15 >= 0.5
    tags.append({
        "label": "+1,5 BUT" if over else "-1,5 BUT",
        "p": p_over15 if over else 1 - p_over15,
        "ok": (hg + ag >= 2 if over else hg + ag <= 1) if played else None,
    })

    result = None
    if played:
        result = 0 if hg > ag else (1 if hg == ag else 2)
    tags.append({
        "label": ["1", "N", "2"][pick],
        "p": float(p1x2[pick]),
        "ok": (result == pick) if played else None,
    })

    gg = p_btts >= 0.5
    tags.append({
        "label": "GG" if gg else "NG",
        "p": p_btts if gg else 1 - p_btts,
        "ok": ((hg > 0 and ag > 0) if gg else not (hg > 0 and ag > 0)) if played else None,
    })

    return {
        "league": league, "date": date_str, "home": home, "away": away,
        "hg": hg, "ag": ag, "tags": tags, "score_pred": f"{i}-{j}",
    }


# ----------------------------------------------------------------------
# Dessin
# ----------------------------------------------------------------------
CARD_H = 560


def _initials(name):
    letters = "".join(w[0] for w in name.replace("-", " ").split() if w)
    return (letters[:3] if len(letters) >= 2 else name[:3]).upper()


def _color(name):
    return PALETTE[sum(ord(c) for c in name) % len(PALETTE)]


def _draw_match(img, y, m, show_prob=True):
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([40, y, W - 40, y + CARD_H], radius=36, fill=CARD)
    _text(d, (W // 2, y + 44), f"{m['league']} · {m['date']}", 30, MUTED)

    for cx, name in ((230, m["home"]), (850, m["away"])):
        d.ellipse([cx - 70, y + 80, cx + 70, y + 220], fill=_color(name))
        _text(d, (cx, y + 150), _initials(name), 48, TXT)
        lines, size = _fit_lines(d, name, 340, 44)
        for k, line in enumerate(lines):
            _text(d, (cx, y + 262 + k * (size + 8)), line, size, TXT)

    if m["hg"] is not None:
        _text(d, (W // 2, y + 150), f"{m['hg']} - {m['ag']}", 130, TXT)
    else:
        _text(d, (W // 2, y + 140), "VS", 96, MUTED)
        _text(d, (W // 2, y + 225), f"score probable {m['score_pred']}", 28, MUTED)

    pill_w, gap = 300, 20
    x = (W - (3 * pill_w + 2 * gap)) // 2
    top = y + CARD_H - 150
    for t in m["tags"]:
        color = BLUE if t["ok"] is None else (GREEN if t["ok"] else RED)
        d.rounded_rectangle([x, top, x + pill_w, top + 112], radius=26, fill=color)
        _text(d, (x + pill_w // 2, top + (42 if show_prob else 56)), t["label"], 50, TXT)
        if show_prob:
            _text(d, (x + pill_w // 2, top + 88), f"{t['p'] * 100:.0f} %", 28, TXT)
        x += pill_w + gap


def render_card(matches, title, signature="", show_prob=True):
    """Image PNG (octets) avec 1 à 4 matchs empilés."""
    n = len(matches)
    head, foot = 170, 150
    height = head + n * (CARD_H + 30) + foot
    img = Image.new("RGB", (W, height), BG)
    d = ImageDraw.Draw(img)

    lines, size = _fit_lines(d, title, W - 120, 64, minimum=36)
    for k, line in enumerate(lines):
        _text(d, (W // 2, 70 + k * (size + 6)), line, size, TXT)

    for k, m in enumerate(matches):
        _draw_match(img, head + k * (CARD_H + 30), m, show_prob)

    y = height - foot + 40
    _text(d, (W // 2, y), "Pronostics statistiques · aucun gain garanti · 18+", 26, MUTED)
    if signature.strip():
        _text(d, (W // 2, y + 50), signature.strip(), 34, TXT)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
