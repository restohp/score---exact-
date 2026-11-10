import importlib

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Auto-correction", page_icon="🔁", layout="wide")

# Vérification des fichiers : dit lequel est manquant ou ancien (au lieu d'une erreur obscure)
_NEEDED = {
    "ac_data": ["load_data", "oos_predictions", "league_share"],
    "ac_fit": ["fit_full", "evaluate_team", "evaluate"],
    "ac_review": ["review_table", "evaluate_half", "half_hit_rates"],
    "calib_core": ["load_data", "fit_full", "review_table"],
    "calib_store": ["predictions", "calibration"],
}
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

import calib_core as cb
import calib_store as store
from dixon_coles import LEAGUES

st.title("🔁 Auto-correction des prédictions")
st.caption(
    "L'appli refait des prédictions dans le passé (chaque match est prédit avec les seuls matchs "
    "qui le précèdent), les compare aux vrais scores finaux et aux scores de mi-temps, repère où "
    "elle se trompe (en moyenne, équipe par équipe, mi-temps par mi-temps) et apprend des "
    "corrections. Chaque correction est jugée sur des matchs qu'elle n'a jamais vus et n'est "
    "utilisée sur les cartes que si elle améliore nettement le résultat. "
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
tab_match, tab_obs, tab_teams, tab_half, tab_eval = st.tabs(
    ["Match par match", "Où l'appli se trompe", "Équipes mal évaluées", "Mi-temps",
     "La correction aide-t-elle ?"]
)


def pct(s):
    return (s * 100).round(1).astype(str) + " %"


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
    st.caption("✔ = l'appli avait raison, ✘ = elle s'est trompée. Pour les mi-temps, le score doit "
               "être exact. Chaque prédiction n'utilise que les matchs précédents.")

# ----------------------------------------------------------------------
with tab_obs:
    table, n = cb.predicted_vs_observed(P)
    st.caption(f"{n} matchs prédits dans le passé. « Prévu » = moyenne des probabilités de l'appli, "
               "« Réel » = fréquence observée.")
    show = table.copy()
    show["Prévu"] = pct(show["Prévu"])
    show["Réel"] = pct(show["Réel"])
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
    rel["Prévu"] = pct(rel["prevu"])
    rel["Réel"] = pct(rel["reel"])
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
with tab_half:
    hh = cb.half_hit_rates(P)
    if hh is None:
        st.info("Ce championnat n'a pas assez de scores de mi-temps dans les fichiers gratuits.")
    else:
        table, n = hh
        st.markdown("**Les scores de mi-temps prédits sur les cartes**")
        st.caption(f"{n} matchs. Le score de chaque mi-temps est celui des cartes : déduit du "
                   "score final probable et de la part de buts de chaque équipe en 1re mi-temps. "
                   "« Toujours le même choix » = réussite si on répondait toujours le score "
                   "(ou le résultat) le plus fréquent : c'est la barre à dépasser.")
        show = table.copy()
        show["Réussite de l'appli"] = pct(show["Réussite de l'appli"])
        show["Toujours le même choix"] = pct(show["Toujours le même choix"])
        st.dataframe(show, hide_index=True, use_container_width=True)

        ev = cb.evaluate_half(P)
        if ev is not None:
            tab, s_lg, nb = ev
            st.markdown("**Quelle part de buts en 1re mi-temps prédit le mieux ?**")
            st.caption(f"Dans ce championnat, {cb.league_share(P):.1%} des buts tombent en 1re "
                       f"mi-temps. Les trois méthodes sont jugées sur les {nb} matchs les plus "
                       "récents (chiffres = log-loss des buts, plus petit = meilleur ; "
                       "z sous -2 = amélioration nette contre la part fixe 45 %).")
            show = tab.copy()
            for c in show.columns[1:]:
                show[c] = show[c].round(4) if not c.startswith("z") else show[c].round(2)
            st.dataframe(show, hide_index=True, use_container_width=True)
            best = []
            for _, r in tab.iterrows():
                if r["z par équipe"] < -2:
                    best.append(f"{r['Mi-temps']} : la part par équipe (carte) est nettement meilleure")
                elif r["z championnat"] < -2:
                    best.append(f"{r['Mi-temps']} : la part du championnat est nettement meilleure")
            if best:
                st.success(" ; ".join(best) + ".")
            else:
                st.info("Aucune méthode ne bat nettement la part fixe de 45 % : les buts par "
                        "mi-temps restent surtout du hasard.")
            st.caption("Sur la carte, la case « Auto-correction » utilise la part réelle du "
                       "championnat comme point de départ du score de chaque mi-temps.")

# ----------------------------------------------------------------------
with tab_eval:
    res, cal, vm, nb = cb.evaluate(P)
    st.markdown("**Correction globale** (trop de victoires à domicile, pas assez de nuls...)")
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
        st.success("La correction globale améliore nettement : " + ", ".join(wins) + ".")
    else:
        st.info("Aucune amélioration nette sur ces matchs : la correction globale est inutile ici, "
                "mais elle ne fait pas de mal non plus.")

    # --- correction par équipe ---
    st.markdown("**Correction équipe par équipe** (attaque et défense de chaque équipe)")
    try:
        tt, z_all, nt = cb.evaluate_team(P)
        show = tt.copy()
        for c in show.columns[1:5]:
            show[c] = show[c].round(4)
        show[show.columns[5]] = show[show.columns[5]].round(2)
        st.dataframe(show, hide_index=True, use_container_width=True)
        st.caption(f"Jugée sur les {nt} matchs les plus récents, avec des corrections apprises "
                   "uniquement sur les 60 % les plus anciens. Chaque équipe est corrigée "
                   "prudemment (ramenée vers « aucune correction » quand on la connaît peu). "
                   "Dans la dernière colonne, un z sous -2 veut dire que la correction par "
                   "équipe aide vraiment.")
        if np.isfinite(z_all) and z_all < -2:
            st.success(f"La correction par équipe améliore nettement l'ensemble des paris "
                       f"(z global {z_all:.2f}) : elle est activée sur les cartes avec "
                       "l'auto-correction.")
        elif np.isfinite(z_all) and z_all > 2:
            st.warning(f"La correction par équipe dégrade le résultat (z global {z_all:.2f}) : "
                       "elle n'est pas utilisée. Les écarts par équipe vus plus haut ne se "
                       "répètent pas assez d'un match à l'autre.")
        else:
            st.info(f"Pas de différence nette (z global {z_all:.2f}) : la correction par équipe "
                    "n'est pas utilisée sur les cartes.")
    except Exception as exc:
        st.caption(f"Évaluation par équipe indisponible ({exc}).")

    if vm:
        st.markdown("**Comparaison avec le marché (résultat 1/N/2)**")
        d1, d2, d3 = st.columns(3)
        d1.metric("Avant correction", f"{vm['avant']:.4f}")
        d2.metric("Après correction", f"{vm['apres']:.4f}")
        d3.metric("Marché (sans marge)", f"{vm['marche']:.4f}")
        st.caption("Même après correction, le marché reste la référence : il connaît les blessures "
                   "et les compositions, pas l'appli.")

    st.markdown("**Correction finale utilisée sur les cartes**")
    full = store.calibration(code)
    used = full.get("used", {})
    a, bn, b2 = full["x12"]
    st.write(("- 1/N/2 : correction utilisée " if used.get("x12") else "- 1/N/2 : non utilisée ")
             + (f"(température {a:.2f}, biais nul {bn:+.2f}, extérieur {b2:+.2f})."
                if used.get("x12") else "(elle n'améliore pas nettement)."))
    st.write(("- +1,5 but : correction utilisée " if used.get("o15") else "- +1,5 but : non utilisée ")
             + (f"(pente {full['o15'][0]:.2f}, décalage {full['o15'][1]:+.2f})."
                if used.get("o15") else "(elle n'améliore pas nettement)."))
    st.write(("- Les deux marquent : correction utilisée " if used.get("gg") else "- Les deux marquent : non utilisée ")
             + (f"(pente {full['gg'][0]:.2f}, décalage {full['gg'][1]:+.2f})."
                if used.get("gg") else "(elle n'améliore pas nettement)."))
    st.write(f"- Part des buts en 1re mi-temps du championnat : {full['share']:.1%}.")
    if full.get("team"):
        st.write("- Correction par équipe : activée. Équipes les plus modifiées (en % de buts) :")
        ft = cb.team_factors_table(full["team"]).round(1)
        st.dataframe(ft, hide_index=True, use_container_width=True)
    else:
        st.write("- Correction par équipe : non activée (elle n'a pas passé le test).")
    st.caption("Sur la page « cartes », la case « Auto-correction » applique seulement les "
               "corrections marquées « utilisée ». Les autres pourcentages restent ceux du modèle.")
