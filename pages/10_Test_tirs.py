import importlib

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Test des tirs", page_icon="🎯", layout="wide")

_NEEDED = {"shots_model": ["run_experiment", "evaluate_experiment", "shots_coverage",
                           "xg_coefficients"],
           "forme_core": ["load_stats"],
           "dixon_coles": ["LEAGUES"]}
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

import forme_core as fc
import shots_model as sm
from dixon_coles import LEAGUES

st.title("🎯 Test : les tirs améliorent-ils le modèle ?")
st.caption(
    "Le modèle actuel estime la force des équipes avec les buts seulement, et les buts sont très "
    "aléatoires. Ce test lui donne aussi les tirs (surtout les tirs cadrés), qui contiennent plus "
    "d'information par match. Chaque match est prédit avec les seuls matchs qui le précèdent, "
    "puis les versions sont comparées sur le même jeu de matchs. "
    "z sous -2 = amélioration nette ; z au-dessus de +2 = dégradation nette."
)

codes = st.multiselect("Championnats à tester", list(LEAGUES), default=["E0"],
                       format_func=lambda c: f"{LEAGUES[c]} ({c})", key="shots_codes")
st.caption("Compte 1 à 3 minutes par championnat. Teste-en plusieurs : un résultat qui ne se répète "
           "pas d'un championnat à l'autre n'est pas fiable.")

if "shots_res" not in st.session_state:
    st.session_state["shots_res"] = {}

if st.button("Lancer le test", type="primary", use_container_width=True, key="shots_go") and codes:
    for code in codes:
        bar = st.progress(0.0, text=f"{LEAGUES[code]} : calcul en cours...")
        try:
            df = fc.load_stats(code, n_seasons=5)
            cov = sm.shots_coverage(df)
            P = sm.run_experiment(df, progress=lambda f, b=bar, c=code: b.progress(
                min(float(f), 1.0), text=f"{LEAGUES[c]} : calcul en cours..."))
            tab, mk, n = sm.evaluate_experiment(P)
            a, b = sm.xg_coefficients(df)
            st.session_state["shots_res"][code] = {"tab": tab, "mk": mk, "n": n, "cov": cov,
                                                   "coef": (a, b)}
        except Exception as exc:
            st.session_state["shots_res"][code] = {"error": str(exc)}
        bar.empty()

res = st.session_state["shots_res"]
if not res:
    st.info("Choisis des championnats puis touche « Lancer le test ».")
    st.stop()


def verdict(z):
    if not np.isfinite(z):
        return "n/a"
    if z < -2:
        return "mieux"
    if z > 2:
        return "moins bien"
    return "pas de différence nette"


# ----------------------------------------------------------------------
st.subheader("Résumé")
rows = []
for code, r in res.items():
    if "error" in r:
        rows.append({"Championnat": LEAGUES[code], "Matchs": "-", "Version": r["error"],
                     "z 1/N/2": "", "Verdict": "", "Écart au marché (z)": ""})
        continue
    tab, mk = r["tab"], r["mk"]
    for i, t in tab.iterrows():
        if i == 0:
            continue
        w = [1.0, 0.5, 0.0][i]
        rows.append({"Championnat": LEAGUES[code], "Matchs": r["n"], "Version": t["Version"],
                     "z 1/N/2": round(float(t["z 1/N/2"]), 2), "Verdict": verdict(t["z 1/N/2"]),
                     "Écart au marché (z)": round(mk["z"][w], 2) if mk else ""})
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
st.caption("« Écart au marché (z) » : z négatif = le modèle fait mieux que les cotes sans marge, "
           "positif = moins bien. Même un z négatif ne suffit pas à gagner de l'argent, car les "
           "bookmakers prennent une marge d'environ 5 % sur les cotes.")

wins = [(LEAGUES[c], r["tab"].iloc[i]["Version"]) for c, r in res.items() if "error" not in r
        for i in (1, 2) if r["tab"].iloc[i]["z 1/N/2"] < -2]
n_ok = sum(1 for r in res.values() if "error" not in r)
if wins:
    st.success("Amélioration nette sur le 1/N/2 : " + " ; ".join(f"{a} ({b})" for a, b in wins) + ".")
else:
    st.info(f"Aucune amélioration nette du 1/N/2 sur {n_ok} championnat(s) testé(s).")

# ----------------------------------------------------------------------
st.subheader("Détail par championnat")
for code, r in res.items():
    with st.expander(f"{LEAGUES[code]} ({code})"):
        if "error" in r:
            st.warning(r["error"])
            continue
        st.caption(f"{r['n']} matchs prédits dans le passé. Tirs disponibles pour {r['cov']:.0%} des "
                   f"matchs. Un tir cadré vaut environ {r['coef'][0]:.2f} but et un tir non cadré "
                   f"environ {r['coef'][1]:.2f} but dans ce championnat.")
        show = r["tab"].copy()
        for c in ("1/N/2", "+1,5 but", "GG"):
            show[c] = show[c].round(4)
        for c in ("z 1/N/2", "z +1,5", "z GG"):
            show[c] = show[c].round(2)
        st.dataframe(show, hide_index=True, use_container_width=True)
        st.caption("Chiffres = log-loss (plus petit = meilleur). z = écart contre le modèle actuel.")
        if r["mk"]:
            mk = r["mk"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Marché (sans marge)", f"{mk['marche']:.4f}")
            m2.metric("Buts seuls", f"{mk['versions'][1.0]:.4f}")
            m3.metric("Mélange", f"{mk['versions'][0.5]:.4f}")
            m4.metric("Tirs seuls", f"{mk['versions'][0.0]:.4f}")
            st.caption(f"Comparaison au marché sur {mk['n']} matchs avec cotes (log-loss du 1/N/2).")

st.caption("Ce test dit seulement si les tirs aident le modèle, pas si le modèle bat les bookmakers. "
           "Une amélioration doit se retrouver dans plusieurs championnats pour être crédible.")
