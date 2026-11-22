import html
import importlib

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Modèle pro", page_icon="🚀", layout="wide")

_NEEDED = {"pro_core": ["build", "fit_model", "predict"],
           "shots_model": ["DixonColesShots", "shots_coverage"],
           "ac_data": ["load_data"], "ac_fit": ["fit_full", "correct", "adjusted_matrix"],
           "calendrier_core": ["collect", "cutoff_for", "fr_date"],
           "cartes_core": ["build_match", "render_card"], "half_stats": ["load_half", "averages"],
           "fixtures": ["load_fixtures"], "dixon_coles": ["LEAGUES"]}
_problems = []
for _mod, _names in _NEEDED.items():
    try:
        _m = importlib.import_module(_mod)
    except Exception as _exc:
        _problems.append(f"{_mod}.py : introuvable ou erreur ({_exc})")
        continue
    for _n in _names:
        if not hasattr(_m, _n):
            _problems.append(f"{_mod}.py : ancienne version (il manque « {_n} »)")
if _problems:
    st.error("Des fichiers sont à corriger sur GitHub (racine du dépôt) :\n\n- " + "\n- ".join(_problems))
    st.stop()

import calendrier_core as cc
import cartes_core as cr
import half_stats as hs
import pro_core as pc
from dixon_coles import LEAGUES
from fixtures import load_fixtures

st.title("🚀 Modèle pro")
st.caption(
    "Le modèle qui réunit tout ce qui a été testé et validé : force des équipes apprise avec les "
    "buts ET les tirs, puis auto-correction des probabilités là où elle a fait ses preuves. "
    "Rien n'est appliqué sans test : chaque amélioration est jugée sur des matchs qu'elle n'a "
    "jamais vus. Tes autres pages ne sont pas modifiées."
)

today = pd.Timestamp.today().normalize()


@st.cache_data(ttl=43200, show_spinner=False)
def pro_build(code):
    return pc.build(code)


@st.cache_data(ttl=43200, show_spinner=False)
def half_df(code):
    return hs.load_half(code, n_seasons=2)


@st.cache_data(ttl=3600, show_spinner=False)
def upcoming():
    return load_fixtures()


def pct(x):
    return f"{x * 100:.1f} %"


code = st.selectbox("Championnat", list(LEAGUES), index=list(LEAGUES).index("E0"),
                    format_func=lambda c: f"{LEAGUES[c]} ({c})", key="pro_code")
day = st.date_input("Jour des matchs", value=today.date(), format="DD/MM/YYYY", key="pro_day")

if st.button("Charger le modèle pro", type="primary", use_container_width=True, key="pro_go"):
    d = day[0] if isinstance(day, (tuple, list)) and day else day
    st.session_state["pro_params"] = (code, pd.Timestamp(d))
    st.session_state.pop("pro_png", None)

params = st.session_state.get("pro_params")
if not params:
    st.info("Choisis un championnat et un jour, puis touche « Charger le modèle pro ». "
            "La première fois, compte 2 à 5 minutes (le modèle s'entraîne et se teste dans le passé).")
    st.stop()

code_run, day_ts = params
try:
    with st.spinner("Entraînement et test du modèle pro (2 à 5 minutes la première fois)..."):
        res = pro_build(code_run)
except Exception as exc:
    st.error(f"Impossible de construire le modèle pour ce championnat ({exc}).")
    st.stop()

df, cal, info = res["df"], res["cal"], res["info"]
used = cal.get("used", {})

# ----------------------------------------------------------------------
st.subheader(f"{LEAGUES[code_run]} : ce que le modèle pro utilise")
z = info["z_shots"]
if info["cov"] < 0.7:
    shots_txt = "non : pas assez de données de tirs dans les fichiers gratuits"
elif info["w"] < 1.0:
    shots_txt = f"oui (mélange 50 % buts, 50 % tirs ; z {z:+.2f} contre les buts seuls, négatif = mieux)"
else:
    shots_txt = f"non : le test les trouve nettement moins bons ici (z {z:+.2f})"
rows = [
    ("Force des équipes avec les tirs", shots_txt),
    ("Correction du 1/N/2", "oui" if used.get("x12") else "non (elle n'améliore pas nettement)"),
    ("Correction du +1,5 but", "oui" if used.get("o15") else "non (elle n'améliore pas nettement)"),
    ("Correction des deux équipes marquent",
     "oui" if used.get("gg") else "non (elle n'améliore pas nettement)"),
    ("Correction équipe par équipe", "oui" if cal.get("team") else "non (jamais validée)"),
    ("Part des buts en 1re mi-temps", pct(cal.get("share", 0.45))),
]
st.dataframe(pd.DataFrame(rows, columns=["Amélioration", "Utilisée ?"]), hide_index=True,
             use_container_width=True)

c1, c2, c3 = st.columns(3)
c1.metric("Modèle actuel (buts seuls)", f"{info['ll_goals']:.4f}")
c2.metric("Modèle pro (brut)", f"{info['ll_blend']:.4f}" if np.isfinite(info["ll_blend"])
          else f"{info['ll_goals']:.4f}")
if "ll_market" in info:
    c3.metric("Marché (sans marge)", f"{info['ll_market']:.4f}")
st.caption(f"Log-loss du 1/N/2 sur {info['n']} matchs prédits dans le passé (plus petit = meilleur). "
           "Si le modèle pro ne bat pas le marché, c'est normal : le marché connaît les blessures et "
           "les compositions. Le modèle pro ne fait que mieux que l'ancien, dans les limites testées.")

# ----------------------------------------------------------------------
st.subheader(f"Matchs du {cc.fr_date(day_ts)}")
fixtures_df = upcoming()
try:
    cutoff = cc.cutoff_for(df, day_ts)
    model = pc.fit_model(df, info["w"], cutoff)
    rows_m, _ = cc.collect(code_run, df, fixtures_df, model, day_ts, day_ts)
except Exception as exc:
    st.error(f"Impossible de lister les matchs ({exc}).")
    st.stop()

if not rows_m:
    st.info("Aucun match ce jour-là pour ce championnat. Les matchs à venir ne couvrent qu'une "
            "à deux semaines.")
    st.stop()

rows_m.sort(key=lambda r: r["home"])
for r in rows_m:
    r["pro"] = pc.predict(model, cal, r["home"], r["away"])

CSS = """
<style>
.mrow{display:flex;align-items:center;gap:10px;padding:10px 8px;border-bottom:1px solid rgba(128,128,128,.25)}
.mrow .tm{flex:1 1 40%;min-width:0}
.mrow .lg{font-size:.72rem;opacity:.65}
.mrow .nm{font-weight:600;line-height:1.25;overflow:hidden;text-overflow:ellipsis}
.mrow .pr{flex:0 0 auto;display:flex;gap:8px;font-size:.95rem;opacity:.75}
.mrow .pr b{opacity:1}
.mrow .bd{flex:0 0 auto;text-align:center;min-width:54px}
.mrow .badge{display:inline-block;width:34px;height:34px;line-height:34px;border-radius:50%;
 background:#e6b422;color:#1a1a1a;font-weight:700}
.mrow .badge.ok{background:#22a35a;color:#fff}
.mrow .badge.ko{background:#d64545;color:#fff}
.mrow .sc{font-size:.9rem;margin-top:2px}
.mrow .od{flex:0 0 46px;text-align:right;font-size:.85rem;opacity:.8}
.mrow .sub{font-size:.72rem;opacity:.6;margin-top:2px}
</style>
"""
LAB = ("1", "X", "2")


def row_html(r):
    s = r["pro"]
    p = [int(round(v * 100)) for v in s["p"]]
    pick = s["pick"]
    pr = "".join(f"<b>{v}</b>" if k == pick else f"<span>{v}</span>" for k, v in enumerate(p))
    if r["hg"] is not None:
        cls = "ok" if r["result"] == pick else "ko"
        score, sub = f"{r['hg']}-{r['ag']}", f"prévu {s['score'][0]}-{s['score'][1]}"
    else:
        cls, score, sub = "", f"{s['score'][0]}-{s['score'][1]}", ""
    odds = r.get("odds")
    od = f"{odds[pick]:.2f}" if odds else "-"
    extra = f"+1,5 : {s['p15']:.0%} · GG : {s['pgg']:.0%}"
    return (
        f'<div class="mrow"><div class="tm"><div class="lg">{html.escape(r["ligue"])}</div>'
        f'<div class="nm">{html.escape(r["home"])}</div><div class="nm">{html.escape(r["away"])}</div>'
        f'<div class="sub">{extra}</div></div><div class="pr">{pr}</div>'
        f'<div class="bd"><div class="badge {cls}">{LAB[pick]}</div>'
        f'<div class="sc">{score}</div><div class="sub">{sub}</div></div>'
        f'<div class="od">{od}</div></div>'
    )


played = [r for r in rows_m if r["hg"] is not None]
if played:
    w = sum(1 for r in played if r["result"] == r["pro"]["pick"])
    st.metric("Pronostics 1/N/2 justes (matchs joués)", f"{w}/{len(played)}",
              f"{w / len(played):.0%}", delta_color="off")
st.markdown(CSS + "".join(row_html(r) for r in rows_m), unsafe_allow_html=True)
st.caption("Probabilités en % de 1 (domicile), X (nul) et 2 (extérieur), corrigées si la correction est "
           "utilisée dans ce championnat ; le pronostic et le score prévu en découlent. "
           "Cote du marché pour ce pronostic si disponible. Ce n'est pas un conseil de pari.")

# ----------------------------------------------------------------------
st.subheader("Carte pronostic (modèle pro)")
labels = []
for r in rows_m:
    score = f" ({r['hg']}-{r['ag']})" if r["hg"] is not None else ""
    labels.append(f"{r['home']} - {r['away']}{score}")
choix = st.multiselect("Matchs à mettre sur la carte (4 maximum)", labels, max_selections=4, key="pro_choix")
title = st.text_input("Titre", value=f"Pronostics du {cc.fr_date(day_ts)}", key="pro_title")
signature = st.text_input("Ta signature (pseudo, optionnel)", value="", key="pro_sig")
show_prob = st.checkbox("Afficher les pourcentages", value=True, key="pro_prob")
show_half = st.checkbox("Afficher les buts par mi-temps", value=True, key="pro_half")

if st.button("Créer la carte", use_container_width=True, key="pro_card"):
    if not choix:
        st.warning("Choisis au moins un match.")
    else:
        hd = half_df(code_run)
        matches = []
        for lab in choix:
            r = rows_m[labels.index(lab)]
            stats = (hs.averages(hd, r["home"], r["date"]), hs.averages(hd, r["away"], r["date"]))
            matches.append(cr.build_match(model, r["home"], r["away"], r["ligue"], cc.fr_date(r["date"]),
                                          r["hg"], r["ag"], stats=stats, calib=cal))
        st.session_state["pro_png"] = cr.render_card(matches, title, signature, show_prob, show_half)

png = st.session_state.get("pro_png")
if png:
    st.image(png, use_container_width=True)
    st.download_button("Télécharger l'image (PNG)", data=png,
                       file_name=f"pronostics_pro_{day_ts:%Y-%m-%d}.png", mime="image/png",
                       use_container_width=True)
st.caption("Un modèle statistique qui ne bat pas le marché : à présenter comme une analyse, "
           "jamais comme un gain assuré.")
