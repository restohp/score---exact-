import pandas as pd
import streamlit as st

import calendrier_core as cc
import cartes_core as cr
import dixon_coles as dc
from dixon_coles import LEAGUES
from fixtures import load_fixtures

st.set_page_config(page_title="Cartes pronostic", page_icon="🖼️", layout="wide")

st.title("🖼️ Cartes pronostic")
st.caption(
    "Crée une image à partager avec les matchs de ton choix : score ou « VS », "
    "et trois étiquettes tirées du modèle (buts, résultat, les deux équipes marquent). "
    "Bleu = à venir, vert = validé, rouge = raté (une fois le match joué)."
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


# ----------------------------------------------------------------------
# 1. Jour et championnats
# ----------------------------------------------------------------------
with st.form("cartes"):
    day = st.date_input("Jour des matchs", value=today.date(), format="DD/MM/YYYY", key="car_day")
    codes_sel = st.multiselect(
        "Championnats", list(LEAGUES), default=DEFAULT_CODES,
        format_func=lambda c: LEAGUES[c], key="car_codes",
    )
    go = st.form_submit_button("Chercher les matchs", use_container_width=True)

if go:
    d = day[0] if isinstance(day, (tuple, list)) and day else day
    st.session_state["car_params"] = (pd.Timestamp(d), tuple(codes_sel))
    st.session_state.pop("car_png", None)
elif "car_params" not in st.session_state:
    st.session_state["car_params"] = (today, tuple(DEFAULT_CODES))

day_ts, codes = st.session_state["car_params"]
st.markdown(f"**{cc.fr_date(day_ts)}**")

if not codes:
    st.info("Choisis au moins un championnat.")
    st.stop()

# ----------------------------------------------------------------------
# 2. Liste des matchs
# ----------------------------------------------------------------------
fixtures_df = upcoming()
rows, failed = [], []

with st.spinner("Chargement des matchs (le premier chargement peut durer 1 à 2 minutes)..."):
    for code in codes:
        try:
            df = history(code)
            model = model_for(code, str(cc.cutoff_for(df, day_ts).date()))
            r, _ = cc.collect(code, df, fixtures_df, model, day_ts, day_ts)
            rows += r
        except Exception as exc:
            failed.append(f"{LEAGUES[code]} ({exc})")

if failed:
    st.caption("Non chargés : " + " ; ".join(failed))

if not rows:
    st.info(
        "Aucun match ce jour-là pour ces championnats. "
        "Les coupes (Copa del Rey, etc.) ne sont pas dans les fichiers gratuits, "
        "et les matchs à venir ne couvrent qu'une à deux semaines."
    )
    st.stop()

rows.sort(key=lambda r: (r["ligue"], r["home"]))
labels = []
for r in rows:
    score = f" ({r['hg']}-{r['ag']})" if r["hg"] is not None else ""
    labels.append(f"{r['ligue']} · {r['home']} - {r['away']}{score}")

# ----------------------------------------------------------------------
# 3. Choix des matchs et création de l'image
# ----------------------------------------------------------------------
choix = st.multiselect(
    "Matchs à mettre sur la carte (4 maximum)", labels, max_selections=4, key="car_choix"
)
title = st.text_input("Titre", value=f"Pronostics du {cc.fr_date(day_ts)}", key="car_title")
signature = st.text_input("Ta signature (pseudo, optionnel)", value="", key="car_sig")
show_prob = st.checkbox("Afficher les pourcentages", value=True, key="car_prob")

if st.button("Créer la carte", type="primary", use_container_width=True):
    if not choix:
        st.warning("Choisis au moins un match.")
    else:
        matches = []
        for lab in choix:
            r = rows[labels.index(lab)]
            df = history(r["code"])
            model = model_for(r["code"], str(cc.cutoff_for(df, day_ts).date()))
            matches.append(cr.build_match(
                model, r["home"], r["away"], r["ligue"], cc.fr_date(r["date"]),
                r["hg"], r["ag"],
            ))
        st.session_state["car_png"] = cr.render_card(matches, title, signature, show_prob)

png = st.session_state.get("car_png")
if png:
    st.image(png, use_container_width=True)
    st.download_button(
        "Télécharger l'image (PNG)",
        data=png,
        file_name=f"pronostics_{day_ts:%Y-%m-%d}.png",
        mime="image/png",
        use_container_width=True,
    )
    st.caption(
        "Ces pronostics viennent d'un modèle statistique qui ne bat pas le marché : "
        "présente-les comme une analyse, jamais comme un gain assuré. "
        "Le vert et le rouge ne s'affichent que sur des matchs réellement joués."
    )
