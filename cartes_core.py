"""Cartes de pronostic en image (format vertical, pour réseaux sociaux).

Design original : pas de logos ni d'écussons de clubs (droits des marques).
Chaque match montre le score (ou « VS »), et trois étiquettes tirées du modèle :
  - buts : « +1,5 BUT » ou « -1,5 BUT » ;
  - résultat : 1, N ou 2 ;
  - les deux équipes marquent : « GG » (oui) ou « NG » (non) ;
  - (si disponible) moyenne de buts marqués / encaissés de chaque équipe en 1re et 2e mi-temps ;
  - (matchs à venir) score probable de chaque mi-temps, cohérent avec le score final probable.
Couleurs : bleu = à venir, vert = validé, rouge = raté (après le match).
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dixon_coles import DixonColes

W = 1080
HALF_SHARE = 0.45   # part des buts marquée en 1re mi-temps (moyenne des grands championnats)
BG = (15, 23, 42)
CARD = (30, 41, 59)
PANEL = (44, 58, 82)
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
def _share(stats):
    """Part des buts en 1re mi-temps : moyenne des deux équipes, ramenée vers 45 %
    (20 « buts fictifs ») pour éviter les valeurs extrêmes sur 10 matchs."""
    if not stats or not stats[0] or not stats[1]:
        return HALF_SHARE
    out = []
    for st in stats:
        g1 = st["n"] * (st["f1"] + st["a1"])
        g2 = st["n"] * (st["f2"] + st["a2"])
        out.append((g1 + HALF_SHARE * 20) / (g1 + g2 + 20))
    return float(np.mean(out))


def half_scores(i, j, share):
    """Score probable de chaque mi-temps, sachant le score final probable i-j.

    Chaque but tombe en 1re mi-temps avec probabilité `share` : le nombre de buts de la
    1re mi-temps suit une loi binomiale, dont on prend la valeur la plus probable.
    La 2e mi-temps est le reste, donc les deux scores s'additionnent toujours au score final.
    """
    h1 = min(int(np.floor((i + 1) * share)), i)
    a1 = min(int(np.floor((j + 1) * share)), j)
    return (h1, a1), (i - h1, j - a1)


def build_match(model, home, away, league, date_str, hg=None, ag=None, stats=None):
    """Étiquettes et probabilités d'un match, + validation si le score est connu.

    stats : (moyennes équipe domicile, moyennes équipe extérieur) par mi-temps, ou None.
    """
    m = model.score_matrix(home, away)
    n = m.shape[0]
    total_goals = np.add.outer(np.arange(n), np.arange(n))
    p_over15 = float(m[total_goals >= 2].sum())
    p_btts = float(1 - m[0, :].sum() - m[:, 0].sum() + m[0, 0])
    p1x2 = DixonColes.probs_1x2(m)
    pick = int(np.argmax(p1x2))
    # score le plus probable PARMI ceux qui correspondent au pronostic 1/N/2
    ii, jj = np.indices(m.shape)
    mask = (ii > jj) if pick == 0 else ((ii == jj) if pick == 1 else (ii < jj))
    i, j = np.unravel_index(np.argmax(np.where(mask, m, -1.0)), m.shape)

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
        "half": stats if (stats and stats[0] and stats[1]) else None,
        "half_score": None if played else half_scores(int(i), int(j), _share(stats)),
    }


# ----------------------------------------------------------------------
# Dessin
# ----------------------------------------------------------------------
CARD_H = 560
HALF_H = 200     # hauteur ajoutée par le bloc « buts par mi-temps »


def _fmt(x):
    return f"{x:.2f}".replace(".", ",")


def _initials(name):
    letters = "".join(w[0] for w in name.replace("-", " ").split() if w)
    return (letters[:3] if len(letters) >= 2 else name[:3]).upper()


def _color(name):
    return PALETTE[sum(ord(c) for c in name) % len(PALETTE)]


def _card_height(m, show_half):
    return CARD_H + (HALF_H if (show_half and m.get("half")) else 0)


def _draw_match(img, y, m, show_prob=True, show_half=True):
    d = ImageDraw.Draw(img)
    extra = _card_height(m, show_half) - CARD_H
    d.rounded_rectangle([40, y, W - 40, y + CARD_H + extra], radius=36, fill=CARD)
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
        if m.get("half_score"):
            (h1, a1), (h2, a2) = m["half_score"]
            _text(d, (W // 2, y + 268), f"1re MT : {h1}-{a1}", 30, TXT)
            _text(d, (W // 2, y + 306), f"2e MT : {h2}-{a2}", 30, TXT)

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

    if extra:
        pw = 450
        x = (W - (2 * pw + gap)) // 2
        top2 = top + 112 + 20
        for name, st in zip((m["home"], m["away"]), m["half"]):
            d.rounded_rectangle([x, top2, x + pw, top2 + 150], radius=26, fill=PANEL)
            size = 30
            while size > 20 and _width(d, name, size) > pw - 30:
                size -= 2
            _text(d, (x + pw // 2, top2 + 28), name, size, MUTED)
            for k, (lab, f, a_) in enumerate((("1re MT", st["f1"], st["a1"]),
                                              ("2e MT", st["f2"], st["a2"]))):
                _text(d, (x + pw // 2, top2 + 78 + k * 44),
                      f"{lab}   +{_fmt(f)} / -{_fmt(a_)}", 34, TXT)
            x += pw + gap
        _text(d, (W // 2, top2 + 150 + 32),
              "+ buts marqués · - buts encaissés, moyenne par match (10 derniers matchs)", 23, MUTED)


def render_card(matches, title, signature="", show_prob=True, show_half=True):
    """Image PNG (octets) avec 1 à 4 matchs empilés."""
    n = len(matches)
    head, foot = 170, 150
    height = head + sum(_card_height(m, show_half) + 30 for m in matches) + foot
    img = Image.new("RGB", (W, height), BG)
    d = ImageDraw.Draw(img)

    lines, size = _fit_lines(d, title, W - 120, 64, minimum=36)
    for k, line in enumerate(lines):
        _text(d, (W // 2, 70 + k * (size + 6)), line, size, TXT)

    y = head
    for m in matches:
        _draw_match(img, y, m, show_prob, show_half)
        y += _card_height(m, show_half) + 30

    y = height - foot + 40
    _text(d, (W // 2, y), "Pronostics statistiques · aucun gain garanti · 18+", 26, MUTED)
    if signature.strip():
        _text(d, (W // 2, y + 50), signature.strip(), 34, TXT)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
