import numpy as np
import pandas as pd
import streamlit as st

import calendrier_core as cc
import cartes_core as cr
import cup_core as cp
from dixon_coles import DixonColes

st.set_page_config(page_title="Coupes européennes", page_icon="🏆", layout="wide")

st.title("🏆 Coupes européennes")
st.caption(
    "Ligue des champions, Ligue Europa, Ligue Conférence. La force des clubs est calculée "
    "sur les matchs européens des dernières saisons (données ouvertes openfootball). "
    "Tu choisis les deux clubs, l'appli calcule les probabilités."
)


@st.cache_data(ttl=21600, show_spinner=False)
def load():
    return cp.load_cups()


@st.cache_resource(show_spinner=False)
def fitted(n_rows, last_date):
    df, _, _ = load()
    return cp.fit_model(df)


with st.spinner("Chargement des matchs européens (1 minute environ la première fois)..."):
    df, fx, report = load()

if df.empty:
    st.error(
        "Impossible de charger les données européennes pour l'instant "
        "(GitHub ne répond pas). Réessaie dans quelques minutes."
    )
    st.stop()

last = df["Date"].max()
st.caption(
    f"{len(df)} matchs chargés, du {cc.fr_date(df['Date'].min())} au {cc.fr_date(last)}. "
    "Les matchs plus récents que la dernière mise à jour du dépôt ne sont pas pris en compte."
)
with st.expander("Détail des fichiers chargés"):
    st.dataframe(
        pd.DataFrame(report, columns=["Saison", "Compétition", "Matchs", "État"]),
        use_container_width=True, hide_index=True,
    )

with st.spinner("Calcul de la force des clubs..."):
    model = fitted(len(df), str(last.date()))

counts = cp.team_table(df)
teams = list(counts.index)

# ----------------------------------------------------------------------
c1, c2 = st.columns(2)
home = c1.selectbox("Équipe à domicile", teams, index=None, placeholder="Choisis un club",
                    key="cup_home")
away = c2.selectbox("Équipe à l'extérieur", teams, index=None, placeholder="Choisis un club",
                    key="cup_away")
comp = st.radio("Compétition", ["Ligue des champions", "Ligue Europa", "Ligue Conférence"],
                horizontal=True, key="cup_comp")
day = st.date_input("Date du match", value=pd.Timestamp.today().date(),
                    format="DD/MM/YYYY", key="cup_day")

if not home or not away:
    st.info("Choisis les deux clubs pour voir les probabilités.")
    st.stop()
if home == away:
    st.warning("Choisis deux clubs différents.")
    st.stop()

if home not in model.idx or away not in model.idx:
    st.error("Un des deux clubs n'est pas dans le modèle.")
    st.stop()

m = model.score_matrix(home, away)
p1, pn, p2 = DixonColes.probs_1x2(m)
n = m.shape[0]
tg = np.add.outer(np.arange(n), np.arange(n))
over15 = float(m[tg >= 2].sum())
over25 = float(m[tg >= 3].sum())
btts = float(1 - m[0, :].sum() - m[:, 0].sum() + m[0, 0])

st.subheader(f"{home} - {away}")
a, b, c = st.columns(3)
a.metric(f"Victoire {home}", f"{p1:.0%}")
b.metric("Match nul", f"{pn:.0%}")
c.metric(f"Victoire {away}", f"{p2:.0%}")
d, e, f = st.columns(3)
d.metric("+1,5 but", f"{over15:.0%}")
e.metric("+2,5 buts", f"{over25:.0%}")
f.metric("Les deux marquent", f"{btts:.0%}")

flat = [(m[i, j], f"{i}-{j}") for i in range(n) for j in range(n)]
flat.sort(reverse=True)
st.caption("Scores les plus probables : " + " · ".join(f"{s} ({p:.0%})" for p, s in flat[:5]))

few = [t for t in (home, away) if counts.get(t, 0) < 10]
if few:
    st.warning(
        "Peu de matchs européens récents pour : " + ", ".join(few) +
        ". Leur force est donc très incertaine."
    )
st.caption(
    "Le modèle donne l'avantage du terrain à l'équipe à domicile. "
    "En finale ou sur terrain neutre, les probabilités sont à nuancer."
)

# ----------------------------------------------------------------------
st.divider()
st.markdown("#### Carte de pronostic")
title = st.text_input("Titre", value=f"{comp} · {cc.fr_date(pd.Timestamp(day))}", key="cup_title")
signature = st.text_input("Ta signature (optionnel)", value="", key="cup_sig")
show_prob = st.checkbox("Afficher les pourcentages", value=True, key="cup_prob")

if st.button("Créer la carte", type="primary", use_container_width=True):
    match = cr.build_match(model, home, away, comp, cc.fr_date(pd.Timestamp(day)))
    st.session_state["cup_png"] = cr.render_card([match], title, signature, show_prob)

png = st.session_state.get("cup_png")
if png:
    st.image(png, use_container_width=True)
    st.download_button("Télécharger l'image (PNG)", data=png, file_name="pronostic_coupe.png",
                       mime="image/png", use_container_width=True)

st.warning(
    "Ces probabilités viennent d'un modèle statistique qui ne bat pas le marché. "
    "Compare toujours avec les cotes réelles ; ne les présente jamais comme un gain assuré."
)
