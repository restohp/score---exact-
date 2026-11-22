import html
import importlib

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Liste du jour", page_icon="📋", layout="wide")

_NEEDED = {"sim_core": ["predict_row"],
           "calendrier_core": ["fit_before", "cutoff_for", "collect", "fr_date"],
           "dixon_coles": ["load_league", "LEAGUES"],
           "fixtures": ["load_fixtures"]}
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
import dixon_coles as dc
import sim_core as sc
from dixon_coles import LEAGUES
from fixtures import load_fixtures

st.title("📋 Liste du jour")
st.caption(
    "Tous les matchs du jour avec, pour chacun, les probabilités 1 / N / 2 issues de la simulation, "
    "le pronostic et le score prévu. Une fois le match joué, le vrai score s'affiche : vert = "
    "pronostic juste, rouge = raté."
)

DEFAULT_CODES = ["E0", "SP1", "D1", "I1", "F1"]
today = pd.Timestamp.today().normalize()


@st.cache_data(ttl=43200, show_spinner=False)
def history(code):
    return dc.load_league(code, n_seasons=4)


@st.cache_resource(show_spinner=False)
def model_for(code, cutoff_key):
    return cc.fit_before(history(code), pd.Timestamp(cutoff_key))


@st.cache_data(ttl=3600, show_spinner=False)
def upcoming():
    return load_fixtures()


with st.form("liste"):
    day = st.date_input("Jour des matchs", value=today.date(), format="DD/MM/YYYY", key="lst_day")
    codes_sel = st.multiselect("Championnats", list(LEAGUES), default=DEFAULT_CODES,
                               format_func=lambda c: LEAGUES[c], key="lst_codes")
    n_sim = st.select_slider("Matchs simulés par rencontre", options=[2000, 5000, 10000, 20000],
                             value=10000, key="lst_n")
    go = st.form_submit_button("Afficher la liste", use_container_width=True)

if go:
    d = day[0] if isinstance(day, (tuple, list)) and day else day
    st.session_state["lst_params"] = (pd.Timestamp(d), tuple(codes_sel), int(n_sim))
elif "lst_params" not in st.session_state:
    st.session_state["lst_params"] = (today, tuple(DEFAULT_CODES), 10000)

day_ts, codes, n_sim = st.session_state["lst_params"]
st.markdown(f"**{cc.fr_date(day_ts)}**")
if not codes:
    st.info("Choisis au moins un championnat.")
    st.stop()

fixtures_df = upcoming()
rows, failed = [], []
with st.spinner("Simulation des matchs (le premier chargement peut durer 1 à 2 minutes)..."):
    for code in codes:
        try:
            df = history(code)
            model = model_for(code, str(cc.cutoff_for(df, day_ts).date()))
            r, _ = cc.collect(code, df, fixtures_df, model, day_ts, day_ts)
            for x in r:
                m = model.score_matrix(x["home"], x["away"])
                x["sim"] = sc.predict_row(m, n=n_sim, seed=42)
            rows += r
        except Exception as exc:
            failed.append(f"{LEAGUES[code]} ({exc})")

if failed:
    st.caption("Non chargés : " + " ; ".join(failed))
if not rows:
    st.info("Aucun match ce jour-là pour ces championnats. Les matchs à venir ne couvrent "
            "qu'une à deux semaines, et les coupes ne sont pas dans les fichiers gratuits.")
    st.stop()

rows.sort(key=lambda r: (r["ligue"], r["home"]))

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
    s = r["sim"]
    p = [int(round(v * 100)) for v in s["p"]]
    pick = s["pick"]
    pr = "".join(f"<b>{v}</b>" if k == pick else f"<span>{v}</span>" for k, v in enumerate(p))
    played = r["hg"] is not None
    if played:
        cls = "ok" if r["result"] == pick else "ko"
        score = f"{r['hg']}-{r['ag']}"
        sub = f"prévu {s['score'][0]}-{s['score'][1]}"
    else:
        cls = ""
        score = f"{s['score'][0]}-{s['score'][1]}"
        sub = ""
    odds = r.get("odds")
    od = f"{odds[pick]:.2f}" if odds else "-"
    extra = f"+2,5 : {s['p_o25']:.0%} · GG : {s['p_gg']:.0%}"
    return (
        f'<div class="mrow"><div class="tm"><div class="lg">{html.escape(r["ligue"])}</div>'
        f'<div class="nm">{html.escape(r["home"])}</div><div class="nm">{html.escape(r["away"])}</div>'
        f'<div class="sub">{extra}</div></div>'
        f'<div class="pr">{pr}</div>'
        f'<div class="bd"><div class="badge {cls}">{LAB[pick]}</div>'
        f'<div class="sc">{score}</div><div class="sub">{sub}</div></div>'
        f'<div class="od">{od}</div></div>'
    )


statut = st.radio("Afficher", ["Tous", "À venir", "Joués"], horizontal=True, key="lst_statut")
shown = [r for r in rows if statut == "Tous" or (statut == "Joués") == (r["hg"] is not None)]

played = [r for r in rows if r["hg"] is not None]
if played:
    w = sum(1 for r in played if r["result"] == r["sim"]["pick"])
    st.metric("Pronostics 1/N/2 justes (matchs joués)", f"{w}/{len(played)}",
              f"{w / len(played):.0%}", delta_color="off")
    st.caption("Sur peu de matchs, ce pourcentage ne prouve rien.")

st.markdown(CSS + "".join(row_html(r) for r in shown), unsafe_allow_html=True)
st.caption("Colonnes : probabilités en % de 1 (domicile), X (nul) et 2 (extérieur), la plus haute en gras ; "
           "pronostic et score prévu (le score le plus fréquent compatible avec le pronostic) ; "
           "cote du marché pour ce pronostic si elle est disponible. "
           "Un modèle statistique qui ne bat pas le marché : ce n'est pas un conseil de pari.")
