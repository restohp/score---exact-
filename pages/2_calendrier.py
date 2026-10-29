import numpy as np
import pandas as pd
import streamlit as st

import calendrier_core as cc
import dixon_coles as dc
from dixon_coles import LEAGUES
from fixtures import load_fixtures

st.set_page_config(page_title="Calendrier", page_icon="📅", layout="wide")

st.title("📅 Calendrier des matchs")
st.caption(
    "Choisis une période : tu vois les matchs, les probabilités du modèle, "
    "les cotes du marché et, pour les matchs déjà joués, le résultat. "
    "Le modèle n'utilise jamais un match de la période pour se prédire lui-même."
)

DEFAULT_CODES = ["E0", "SP1", "D1", "I1", "F1"]
MAX_DAYS = 31
today = pd.Timestamp.today().normalize()


@st.cache_data(ttl=43200, show_spinner=False)
def history(code):
    return dc.load_league(code, n_seasons=4)


@st.cache_resource(show_spinner=False)
def model_for(code, cutoff_key):
    df = history(code)
    return cc.fit_before(df, pd.Timestamp(cutoff_key))


@st.cache_data(ttl=3600, show_spinner=False)
def upcoming():
    return load_fixtures()


# ----------------------------------------------------------------------
# Choix de la période et des championnats
# ----------------------------------------------------------------------
with st.form("calendrier"):
    choix = st.radio(
        "Période",
        ["Aujourd'hui", "Demain", "7 prochains jours", "7 derniers jours",
         "Choisir les dates"],
        horizontal=True,
        key="cal_choix",
    )
    dates = st.date_input(
        "Dates (utilisées si « Choisir les dates »)",
        value=(today.date(), (today + pd.Timedelta(days=6)).date()),
        format="DD/MM/YYYY",
        key="cal_dates",
    )
    codes_sel = st.multiselect(
        "Championnats",
        list(LEAGUES),
        default=DEFAULT_CODES,
        format_func=lambda c: LEAGUES[c],
        key="cal_codes",
    )
    toutes = st.checkbox("Toutes les ligues (premier chargement plus long)", key="cal_all")
    go = st.form_submit_button("Afficher les matchs", use_container_width=True)


def period_from(choix, dates):
    if choix == "Aujourd'hui":
        return today, today
    if choix == "Demain":
        t = today + pd.Timedelta(days=1)
        return t, t
    if choix == "7 prochains jours":
        return today, today + pd.Timedelta(days=6)
    if choix == "7 derniers jours":
        return today - pd.Timedelta(days=7), today - pd.Timedelta(days=1)
    if isinstance(dates, (tuple, list)):
        if len(dates) == 2:
            return pd.Timestamp(dates[0]), pd.Timestamp(dates[1])
        if len(dates) == 1:
            return pd.Timestamp(dates[0]), pd.Timestamp(dates[0])
    return pd.Timestamp(dates), pd.Timestamp(dates)


if go:
    start, end = period_from(choix, dates)
    if end < start:
        start, end = end, start
    codes = list(LEAGUES) if toutes else list(codes_sel)
    st.session_state["cal_params"] = (start, end, tuple(codes))
elif "cal_params" not in st.session_state:
    st.session_state["cal_params"] = (today, today, tuple(DEFAULT_CODES))

start, end, codes = st.session_state["cal_params"]

if (end - start).days + 1 > MAX_DAYS:
    end = start + pd.Timedelta(days=MAX_DAYS - 1)
    st.warning(f"Période limitée à {MAX_DAYS} jours : affichage jusqu'au {cc.fr_date(end)}.")

if not codes:
    st.info("Choisis au moins un championnat.")
    st.stop()

st.markdown(
    f"**{cc.fr_date(start)}**" + ("" if start == end else f" → **{cc.fr_date(end)}**")
)

# ----------------------------------------------------------------------
# Collecte
# ----------------------------------------------------------------------
fixtures_df = upcoming()
rows, skipped, failed = [], 0, []

with st.spinner("Calcul des prévisions (le premier chargement peut durer 1 à 2 minutes)..."):
    for code in codes:
        try:
            df = history(code)
            model = model_for(code, str(cc.cutoff_for(df, start).date()))
            r, s = cc.collect(code, df, fixtures_df, model, start, end)
            rows += r
            skipped += s
        except Exception as exc:
            failed.append(f"{LEAGUES[code]} ({exc})")

if failed:
    st.caption("Non chargés : " + " ; ".join(failed))

if not rows:
    st.info(
        "Aucun match pour cette période et ces championnats. "
        "Rappel : le fichier des matchs à venir ne couvre qu'une à deux semaines "
        "et certains championnats peuvent être en pause."
    )
    st.stop()

# ----------------------------------------------------------------------
# Bilan et affichage
# ----------------------------------------------------------------------
n_played = sum(r["result"] is not None for r in rows)
a, b, c = st.columns(3)
a.metric("Matchs", len(rows))
b.metric("Joués", n_played)
c.metric("À venir", len(rows) - n_played)

bilan = cc.summarize_played(rows)
if bilan is not None:
    st.markdown("#### Bilan sur les matchs joués")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Modèle : favori juste", f"{bilan['acc_model'] * 100:.0f} %")
    k2.metric("Marché : favori juste", f"{bilan['acc_market'] * 100:.0f} %")
    k3.metric("Erreur Brier modèle", f"{bilan['brier_model']:.3f}")
    k4.metric("Erreur Brier marché", f"{bilan['brier_market']:.3f}")
    st.caption(
        f"Sur {bilan['n']} matchs avec cotes. Brier : plus bas = mieux. "
        "Sur de longues périodes, le marché fait mieux que le modèle ; "
        "sur quelques jours, l'écart ne veut rien dire (trop peu de matchs)."
    )

table = cc.to_table(rows).sort_values(["date", "Ligue", "Match"])
for day, part in table.groupby("date"):
    st.markdown(f"#### {cc.fr_date(day)}")
    st.dataframe(
        part.drop(columns=["date"]),
        use_container_width=True,
        hide_index=True,
    )

if skipped:
    st.caption(
        f"{skipped} match(s) non analysés : équipe absente de l'historique "
        "(promue ou nouvelle)."
    )

st.warning(
    "Ces probabilités viennent d'un modèle statistique qui ne bat pas le marché. "
    "Utilise la colonne « Marché » comme référence : si le modèle s'en écarte beaucoup, "
    "c'est presque toujours le modèle qui se trompe."
)
