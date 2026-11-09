import numpy as np
import pandas as pd
import streamlit as st

import calib_core as cb
import calib_store as store
from dixon_coles import LEAGUES

st.set_page_config(page_title="Auto-correction", page_icon="🔁", layout="wide")

st.title("🔁 Auto-correction des prédictions")
st.caption(
    "L'appli refait des prédictions dans le passé (chaque match est prédit avec les seuls matchs "
    "qui le précèdent), les compare aux vrais scores, repère où elle se trompe en moyenne "
    "(trop de victoires à domicile, pas assez de nuls, trop de matchs à +1,5 but...) "
    "et apprend une correction. La correction est jugée sur des matchs qu'elle n'a jamais vus. "
    "Les vrais scores viennent des fichiers football-data, mis à jour chaque jour."
)

code = st.selectbox("Championnat", list(LEAGUES), index=list(LEAGUES).index("E0"),
                    format_func=lambda c: f"{LEAGUES[c]} ({c})", key="ac_code")

if st.button("Analyser ce championnat", type="primary", use_container_width=True, key="ac_run"):
    with st.spinner("Calcul en cours (1 à 3 minutes la première fois)..."):
        try:
            store.predictions(code)
            st.session_state["ac_done"] = code
        except Exception as exc:
            st.session_state["ac_done"] = None
            st.error(str(exc))

if st.session_state.get("ac_done") != code:
    st.info("Choisis un championnat puis touche « Analyser ce championnat ».")
    st.stop()

P = store.predictions(code)
tab_match, tab_obs, tab_teams, tab_eval = st.tabs(
    ["Match par match", "Où l'appli se trompe", "Équipes mal évaluées", "La correction aide-t-elle ?"]
)

# ----------------------------------------------------------------------
with tab_match:
    rates = cb.hit_rates(P)
    st.caption(f"Sur {len(P)} matchs prédits dans le passé, voici le taux de bons pronostics :")
    c1, c2, c3 = st.columns(3)
    c1.metric("Résultat 1/N/2", f"{rates['Résultat 1/N/2']:.0%}")
    c2.metric("+1,5 but", f"{rates['+1,5 but']:.0%}")
    c3.metric("Les deux marquent", f"{rates['Les deux marquent']:.0%}")
    st.caption("Un pronostic = le choix jugé le plus probable. Pour le 1/N/2, répondre au hasard "
               "donne environ 33 %, et le favori des bookmakers environ 50 %.")
    n_show = st.slider("Nombre de matchs affichés", 10, 100, 40, 10, key="ac_n")
    st.dataframe(cb.review_table(P, n_show), hide_index=True, use_container_width=True)
    st.caption("✔ = l'appli avait raison, ✘ = elle s'est trompée. Chaque prédiction n'utilise "
               "que les matchs précédents.")

# ----------------------------------------------------------------------
with tab_obs:
    table, n = cb.predicted_vs_observed(P)
    st.caption(f"{n} matchs prédits dans le passé. « Prévu » = moyenne des probabilités de l'appli, "
               "« Réel » = fréquence observée.")
    show = table.copy()
    show["Prévu"] = (show["Prévu"] * 100).round(1).astype(str) + " %"
    show["Réel"] = (show["Réel"] * 100).round(1).astype(str) + " %"
    show["Écart (points)"] = show["Écart (points)"].round(1)
    st.dataframe(show, hide_index=True, use_container_width=True)
    st.caption("Écart positif = l'appli en prévoit trop ; négatif = pas assez.")

    st.markdown("**Fiabilité par niveau de probabilité**")
    choice = st.selectbox("Pari", ["Match nul", "Victoire domicile", "+1,5 but", "Les deux marquent"],
                          key="ac_choice")
    col, y = {
        "Match nul": (P["pn"], P["y"] == 1),
        "Victoire domicile": (P["p1"], P["y"] == 0),
        "+1,5 but": (P["p15"], P["o15"]),
        "Les deux marquent": (P["pgg"], P["gg"]),
    }[choice]
    rel = cb.reliability(col, y)
    rel["Prévu"] = (rel["prevu"] * 100).round(1).astype(str) + " %"
    rel["Réel"] = (rel["reel"] * 100).round(1).astype(str) + " %"
    st.dataframe(rel[["Prévu", "Réel", "n"]].rename(columns={"n": "Matchs"}),
                 hide_index=True, use_container_width=True)

# ----------------------------------------------------------------------
with tab_teams:
    te = cb.team_errors(P)
    if te.empty:
        st.info("Pas assez de matchs par équipe.")
    else:
        st.caption("Pour chaque équipe : buts réels moins buts attendus par l'appli, par match. "
                   "Positif en « marqués » = l'appli la sous-estime en attaque. Positif en "
                   "« encaissés » = l'appli la surestime en défense. Un z au-delà de 2 (en valeur "
                   "absolue) signale un écart net ; sur 20 équipes, un ou deux peuvent sortir "
                   "par hasard.")
        show = te.copy()
        for c in ("Buts marqués (réel - prévu)", "Buts encaissés (réel - prévu)", "z marqués", "z encaissés"):
            show[c] = show[c].round(2)
        st.dataframe(show, hide_index=True, use_container_width=True)

# ----------------------------------------------------------------------
with tab_eval:
    res, cal, vm, nb = cb.evaluate(P)
    st.caption(f"La correction est apprise sur les 60 % de matchs les plus anciens, "
               f"puis jugée sur les {nb} matchs les plus récents.")
    show = res.copy()
    for c in ("Avant correction", "Après correction"):
        show[c] = show[c].round(4)
    show["z (négatif = mieux)"] = show["z (négatif = mieux)"].round(2)
    st.dataframe(show, hide_index=True, use_container_width=True)
    st.caption("Chiffres = log-loss (plus petit = meilleur). z sous -2 : amélioration nette, "
               "pas due au hasard.")

    wins = [r["Marché"] for _, r in res.iterrows() if r["z (négatif = mieux)"] < -2]
    if wins:
        st.success("La correction améliore nettement : " + ", ".join(wins) + ".")
    else:
        st.info("Aucune amélioration nette sur ces matchs : la correction est inutile ici, "
                "mais elle ne fait pas de mal non plus.")

    if vm:
        st.markdown("**Comparaison avec le marché (résultat 1/N/2)**")
        d1, d2, d3 = st.columns(3)
        d1.metric("Avant correction", f"{vm['avant']:.4f}")
        d2.metric("Après correction", f"{vm['apres']:.4f}")
        d3.metric("Marché (sans marge)", f"{vm['marche']:.4f}")
        st.caption("Même après correction, le marché reste la référence : il connaît les blessures "
                   "et les compositions, pas l'appli.")

    st.markdown("**Correction apprise (sur tous les matchs)**")
    full = cb.fit_calibration(P)
    a, bn, b2 = full["x12"]
    st.write(f"- 1/N/2 : température {a:.2f} (1 = aucune), biais sur le nul {bn:+.2f}, "
             f"sur la victoire extérieur {b2:+.2f}.")
    st.write(f"- +1,5 but : pente {full['o15'][0]:.2f}, décalage {full['o15'][1]:+.2f}.")
    st.write(f"- Les deux marquent : pente {full['gg'][0]:.2f}, décalage {full['gg'][1]:+.2f}.")
    st.caption("Sur la page « cartes », la case « Auto-correction » applique cette correction.")
