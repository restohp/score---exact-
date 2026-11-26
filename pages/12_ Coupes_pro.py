import numpy as np
import pandas as pd
import streamlit as st

import calendrier_core as cc
import cartes_core as cr
import euro_core as ec
from dixon_coles import DixonColes

st.set_page_config(page_title="Coupes pro", page_icon="🏆", layout="wide")

st.title("🏆 Coupes pro")
st.caption(
    "Ligue des champions, Europa League, Conférence League. L'Elo compare les clubs de pays "
    "différents (qui est favori) ; la ligue d'origine de chaque club (buts marqués et encaissés "
    "sur 1 an) donne le rythme du match. Estimation non testée sur des matchs de coupe passés."
)


@st.cache_data(ttl=21600, show_spinner=False)
def elo_table():
    return ec.fetch_elo()


@st.cache_data(ttl=43200, show_spinner=False)
def league_games(country):
    return ec.fetch_league_games(country)


class _Cache(dict):
    """Cache de ligues branché sur st.cache_data."""

    def __missing__(self, country):
        v = league_games(country)
        self[country] = v
        return v

    def __contains__(self, k):
        return True


try:
    with st.spinner("Chargement du classement Elo des clubs..."):
        elo = elo_table()
except Exception as exc:
    st.error(f"Impossible de charger le classement Elo ({exc}). Réessaie dans quelques minutes.")
    st.stop()

elo = elo[elo["Level"].fillna(9) <= 2]
labels = {r.Club: f"{r.Club} ({r.Country}, Elo {r.Elo:.0f})" for r in elo.itertuples()}
clubs = list(elo["Club"])

c1, c2 = st.columns(2)
home = c1.selectbox("Équipe à domicile", clubs, index=None, placeholder="Tape un club",
                    format_func=lambda c: labels[c], key="cp_home")
away = c2.selectbox("Équipe à l'extérieur", clubs, index=None, placeholder="Tape un club",
                    format_func=lambda c: labels[c], key="cp_away")
comp = st.radio("Compétition", ["Ligue des champions", "Ligue Europa", "Ligue Conférence"],
                horizontal=True, key="cp_comp")
neutral = st.checkbox("Terrain neutre (finale)", value=False, key="cp_neutral")
day = st.date_input("Date du match", value=pd.Timestamp.today().date(),
                    format="DD/MM/YYYY", key="cp_day")

if not home or not away:
    st.info("Choisis les deux clubs pour voir les probabilités.")
    st.stop()
if home == away:
    st.warning("Choisis deux clubs différents.")
    st.stop()

with st.spinner("Récupération des ligues d'origine..."):
    res = ec.analyse(elo, home, away, neutral, _Cache())

m = res["matrix"]
p1, pn, p2 = DixonColes.probs_1x2(m)
n = m.shape[0]
tg = np.add.outer(np.arange(n), np.arange(n))
over15, over25 = float(m[tg >= 2].sum()), float(m[tg >= 3].sum())
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
flat = sorted(((m[i, j], f"{i}-{j}") for i in range(n) for j in range(n)), reverse=True)
st.caption("Scores les plus probables : " + " · ".join(f"{s} ({p:.0%})" for p, s in flat[:5]))
st.caption(f"Buts attendus : {res['lh']:.2f} - {res['la']:.2f}")

st.markdown("**D'où viennent les chiffres**")
for side in ("home", "away"):
    t = res["teams"][side]
    st.markdown(f"- {t['club']} : Elo {t['elo']:.0f} ; {t['source']}")
if any(res["teams"][s]["n"] == 0 for s in ("home", "away")):
    st.warning("Pour un club sans données de ligue, seul l'Elo est utilisé : le nombre de buts "
               "attendu est alors une moyenne de coupe, moins précise.")

# ----------------------------------------------------------------------
st.divider()
st.markdown("#### Carte de pronostic (jusqu'à 4 matchs)")
lst = st.session_state.setdefault("cp_list", [])
b1, b2 = st.columns(2)
if b1.button("Ajouter ce match à la carte", use_container_width=True, key="cp_add"):
    if len(lst) >= 4:
        st.warning("4 matchs maximum sur la carte.")
    else:
        lst.append(cr.build_match(ec.MatrixModel(m), home, away, comp,
                                  cc.fr_date(pd.Timestamp(day))))
        st.session_state.pop("cp_png", None)
if b2.button("Vider la carte", use_container_width=True, key="cp_clear"):
    lst.clear()
    st.session_state.pop("cp_png", None)

for k, mt in enumerate(lst, 1):
    st.write(f"{k}. {mt['home']} - {mt['away']} ({mt['league']})")

title = st.text_input("Titre", value=f"{comp} · {cc.fr_date(pd.Timestamp(day))}", key="cp_title")
signature = st.text_input("Ta signature (optionnel)", value="", key="cp_sig")
show_prob = st.checkbox("Afficher les pourcentages", value=True, key="cp_prob")

if st.button("Créer la carte", type="primary", use_container_width=True, key="cp_card"):
    if not lst:
        st.warning("Ajoute d'abord au moins un match.")
    else:
        st.session_state["cp_png"] = cr.render_card(list(lst), title, signature, show_prob, False)

png = st.session_state.get("cp_png")
if png:
    st.image(png, use_container_width=True)
    st.download_button("Télécharger l'image (PNG)", data=png, file_name="pronostics_coupes.png",
                       mime="image/png", use_container_width=True)

st.warning("Estimation non validée sur des matchs de coupe passés, qui ne bat pas le marché. "
           "À présenter comme une analyse, jamais comme un gain assuré.")
