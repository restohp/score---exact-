import numpy as np
import pandas as pd
import streamlit as st

import calendrier_core as cc
import dixon_coles as dc
import forme_core as fc
from dixon_coles import LEAGUES, DixonColes

st.set_page_config(page_title="Forme et statistiques", page_icon="📊", layout="wide")

st.title("📊 Forme et statistiques")
st.caption(
    "Forme sur les 5, 10 et 15 derniers matchs (les plus récents comptent plus), tirs cadrés, "
    "domicile / extérieur séparés, confrontations directes, et un modèle combiné. "
    "Tout est calculé avec des matchs strictement antérieurs au match analysé."
)


@st.cache_data(ttl=43200, show_spinner=False)
def history(code):
    return fc.load_stats(code, n_seasons=5)


@st.cache_resource(show_spinner=False)
def logs_for(code, last_key):
    return fc.Logs(history(code))


@st.cache_resource(show_spinner=False)
def model_for(code, last_key):
    df = history(code)
    return DixonColes().fit(df, ref_date=df["Date"].max())


@st.cache_data(ttl=43200, show_spinner=False)
def run_backtest(code, last_key):
    bt = fc.backtest(history(code))
    return fc.summarize(bt)


code = st.selectbox("Championnat", list(LEAGUES), index=list(LEAGUES).index("E0"),
                    format_func=lambda c: f"{LEAGUES[c]} ({c})", key="fs_code")

with st.spinner("Chargement des matchs (1 à 2 minutes la première fois)..."):
    try:
        df = history(code)
    except Exception as exc:
        st.error(f"Impossible de charger ce championnat : {exc}")
        st.stop()

last = df["Date"].max()
last_key = str(last.date())
has_shots = df["HST"].notna().mean()
st.caption(f"{len(df)} matchs, jusqu'au {cc.fr_date(last)}. "
           f"Tirs cadrés disponibles pour {has_shots:.0%} des matchs.")

tab_match, tab_test = st.tabs(["Analyser un match", "Test sur l'historique"])

# ----------------------------------------------------------------------
with tab_match:
    logs = logs_for(code, last_key)
    model = model_for(code, last_key)
    recent = df[df["Date"] >= last - pd.Timedelta(days=400)]
    teams = sorted(set(recent["HomeTeam"]) | set(recent["AwayTeam"]))

    c1, c2 = st.columns(2)
    home = c1.selectbox("Domicile", teams, index=None, placeholder="Choisis un club", key="fs_home")
    away = c2.selectbox("Extérieur", teams, index=None, placeholder="Choisis un club", key="fs_away")
    if not home or not away or home == away:
        st.info("Choisis deux clubs différents.")
    else:
        before = last + pd.Timedelta(days=1)

        def form_table(team, venue=None):
            rows = []
            for n in fc.WINDOWS:
                s = logs.stats(team, before, n, venue)
                if s is None:
                    continue
                rows.append({
                    "Derniers matchs": f"{s['matchs']}",
                    "Points/match": round(s["points"], 2),
                    "Buts marqués": round(s["buts_pour"], 2),
                    "Buts encaissés": round(s["buts_contre"], 2),
                    "Tirs cadrés pour": None if np.isnan(s["tc_pour"]) else round(s["tc_pour"], 1),
                    "Tirs cadrés contre": None if np.isnan(s["tc_contre"]) else round(s["tc_contre"], 1),
                    "V-N-D": f"{s['victoires']}-{s['nuls']}-{s['defaites']}",
                })
            return pd.DataFrame(rows)

        for team, venue, lab in [(home, 1, "à domicile"), (away, 0, "à l'extérieur")]:
            st.subheader(team)
            st.markdown("**Tous les matchs** (pondérés : le plus récent compte le plus)")
            st.dataframe(form_table(team), hide_index=True, use_container_width=True)
            st.markdown(f"**Seulement {lab}**")
            st.dataframe(form_table(team, venue), hide_index=True, use_container_width=True)

        st.subheader("Confrontations directes (5 dernières années)")
        h = logs.h2h(home, away, before)
        if h.empty:
            st.caption("Aucune confrontation dans cette période.")
        else:
            show = h[["Date", "HomeTeam", "FTHG", "FTAG", "AwayTeam"]].copy()
            show["Date"] = show["Date"].dt.strftime("%d/%m/%Y")
            show.columns = ["Date", "Domicile", "But D", "But E", "Extérieur"]
            st.dataframe(show, hide_index=True, use_container_width=True)
            st.caption("Ces matchs sont affichés à titre d'information : le modèle ne les utilise pas, "
                       "pour ne pas leur donner trop de poids face aux performances récentes.")

        st.subheader("Probabilités")
        try:
            p_dc = DixonColes.probs_1x2(model.score_matrix(home, away))
        except Exception:
            p_dc = None
        p_fo = logs.form_probs(home, away, before)
        w = st.slider("Poids du modèle Dixon-Coles dans le combiné", 0.0, 1.0, 0.5, 0.05, key="fs_w")
        table = {}
        if p_dc is not None:
            table["Dixon-Coles"] = p_dc
        if p_fo is not None:
            table["Forme récente"] = p_fo
        if p_dc is not None and p_fo is not None:
            table["Combiné"] = w * p_dc + (1 - w) * p_fo

        st.markdown("**Cotes du bookmaker (optionnel, tu peux mettre 3 décimales)**")
        o1, o2, o3 = st.columns(3)
        odd1 = o1.number_input("Cote 1", min_value=1.0, value=None, step=0.001, format="%.3f", key="fs_o1")
        oddn = o2.number_input("Cote N", min_value=1.0, value=None, step=0.001, format="%.3f", key="fs_on")
        odd2 = o3.number_input("Cote 2", min_value=1.0, value=None, step=0.001, format="%.3f", key="fs_o2")
        margin = None
        if odd1 and oddn and odd2 and min(odd1, oddn, odd2) > 1:
            fair, margin = fc.fair_probs([odd1, oddn, odd2])
            table["Marché (sans marge)"] = fair

        if table:
            out = pd.DataFrame(
                {k: [f"{x:.1%}" for x in v] for k, v in table.items()},
                index=[f"Victoire {home}", "Nul", f"Victoire {away}"])
            st.dataframe(out, use_container_width=True)
        if margin is not None:
            st.caption(f"Marge du bookmaker : {margin:.1%}. Si le modèle s'écarte beaucoup du marché, "
                       "c'est presque toujours le modèle qui se trompe (blessures, compositions...).")
        st.caption("Les probabilités du modèle ne tiennent pas compte des blessures, suspensions et compositions.")

# ----------------------------------------------------------------------
with tab_test:
    st.markdown(
        "Le test refait l'histoire : pour chaque match des derniers mois, chaque modèle prédit "
        "en n'utilisant que ce qui était connu **avant** le match, puis on compare au marché. "
        "Le poids du modèle combiné est choisi sur la première moitié du test et jugé sur la seconde."
    )
    if st.button("Lancer le test sur ce championnat", type="primary", use_container_width=True, key="fs_run"):
        with st.spinner("Calcul en cours (1 à 3 minutes)..."):
            try:
                st.session_state["fs_bt"] = (code, run_backtest(code, last_key))
            except Exception as exc:
                st.session_state["fs_bt"] = (code, exc)
    res = st.session_state.get("fs_bt")
    if res and res[0] == code:
        if isinstance(res[1], Exception):
            st.error(str(res[1]))
        else:
            tab, best_w, n = res[1]
            st.caption(f"{n} matchs testés (avec cotes valides).")
            st.dataframe(tab.round(4), hide_index=True, use_container_width=True)
            comb = tab.iloc[2]
            z = comb["z (positif = pire que le marché)"]
            if np.isnan(z) or z > -2:
                st.info("Verdict : le modèle combiné ne fait pas mieux que le marché de façon significative. "
                        "C'est le résultat attendu : le marché intègre déjà la forme, les tirs et les blessures.")
            else:
                st.success("Le combiné fait significativement mieux que le marché sur la 2e moitié du test. "
                           "Avant d'y croire, refais le test sur d'autres championnats : un seul résultat peut être dû au hasard.")
