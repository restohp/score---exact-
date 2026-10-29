import numpy as np
import pandas as pd
import streamlit as st

import corners_model as cm
from dixon_coles import LEAGUES

st.set_page_config(page_title="Corners et cartons", page_icon="🚩", layout="wide")

st.title("🚩 Corners et cartons")
st.caption(
    "Estimation par équipe du nombre de corners ou de cartons d'un match. "
    "Les fichiers gratuits contiennent les résultats, pas les cotes de ces "
    "marchés : cette page mesure la qualité des prévisions, elle ne prouve "
    "aucun avantage face aux bookmakers."
)


@st.cache_data(ttl=43200, show_spinner=False)
def load(code, kind):
    return cm.load_counts(code, kind, n_seasons=5)


@st.cache_resource(show_spinner=False)
def fit_model(code, kind, last_date):
    # last_date sert uniquement de clé : le modèle est refait quand de
    # nouveaux matchs arrivent.
    return cm.CountModel().fit(load(code, kind))


@st.cache_data(ttl=43200, show_spinner=False)
def run_backtest(code, kind, lines):
    return cm.backtest(load(code, kind), list(lines))


c1, c2 = st.columns(2)
code = c1.selectbox(
    "Championnat", list(LEAGUES), format_func=lambda c: LEAGUES[c], key="cc_league"
)
kind = c2.radio("Marché", list(cm.KINDS), horizontal=True, key="cc_kind")
cfg = cm.KINDS[kind]

with st.spinner("Chargement des matchs..."):
    df = load(code, kind)

if df.empty:
    st.warning(
        f"Pas de données « {kind.lower()} » pour {LEAGUES[code]} dans les "
        "fichiers gratuits. Choisis un autre championnat."
    )
    st.stop()

mean_total = float((df["h"] + df["a"]).mean())
st.caption(
    f"{len(df)} matchs du {df['Date'].min():%d/%m/%Y} au {df['Date'].max():%d/%m/%Y} "
    f"· moyenne : {mean_total:.1f} {kind.lower()} par match."
)

try:
    model = fit_model(code, kind, str(df["Date"].max().date()))
except Exception as exc:
    st.error(f"Impossible d'ajuster le modèle : {exc}")
    st.stop()

tab_pred, tab_test = st.tabs(["Prédire un match", "Test de fiabilité"])

# ----------------------------------------------------------------------
with tab_pred:
    a, b = st.columns(2)
    home = a.selectbox("Équipe à domicile", model.active, key=f"cc_home_{code}_{kind}")
    away_options = [t for t in model.active if t != home]
    away = b.selectbox("Équipe à l'extérieur", away_options, key=f"cc_away_{code}_{kind}")

    lines = st.multiselect(
        "Lignes à afficher", cfg["lines"], default=cfg["default"], key=f"cc_lines_{kind}"
    )

    if not lines:
        st.info("Choisis au moins une ligne.")
    else:
        lines = sorted(lines)
        res = model.predict(home, away, lines)

        m1, m2, m3 = st.columns(3)
        m1.metric(f"{kind} attendus — {home}", f"{res['home']:.1f}")
        m2.metric(f"{kind} attendus — {away}", f"{res['away']:.1f}")
        m3.metric("Total attendu", f"{res['total']:.1f}")

        t = res["table"]
        shown = pd.DataFrame({
            "Ligne": t["ligne"],
            "Plus (%)": (t["p_plus"] * 100).round(1),
            "Cote juste plus": t["cote_juste_plus"].round(2),
            "Moins (%)": (t["p_moins"] * 100).round(1),
            "Cote juste moins": t["cote_juste_moins"].round(2),
        })
        st.dataframe(shown, use_container_width=True, hide_index=True)
        st.caption(
            "La cote juste est l'inverse de la probabilité du modèle, sans marge. "
            "Un bookmaker affiche toujours une cote plus basse que la cote juste "
            "du marché : pour qu'un pari ait de la valeur, il faudrait que la "
            "cote proposée dépasse la cote juste VRAIE, pas seulement celle de ce modèle."
        )

        with st.expander("Comparer à la cote d'un bookmaker"):
            line = st.selectbox("Ligne", lines, key=f"cc_cmp_line_{kind}")
            x, y = st.columns(2)
            o_plus = x.number_input(
                "Cote « plus »", min_value=0.0, value=0.0, step=0.01,
                format="%.2f", key=f"cc_o_plus_{kind}",
            )
            o_moins = y.number_input(
                "Cote « moins »", min_value=0.0, value=0.0, step=0.01,
                format="%.2f", key=f"cc_o_moins_{kind}",
            )
            st.caption("Laisse 0 pour une cote que tu n'as pas. Avec les deux cotes, la marge est retirée.")

            if o_plus > 1 or o_moins > 1:
                p_over = model.over_prob(res["total"], line)
                r = cm.compare_to_odds(
                    p_over, o_plus if o_plus > 1 else None, o_moins if o_moins > 1 else None
                )
                if "marge_pct" in r:
                    st.caption(f"Marge du bookmaker sur ce marché : {r['marge_pct']:.1f} %.")
                for side, label in (("plus", "Plus"), ("moins", "Moins")):
                    if f"implicite_{side}" in r:
                        p_model = p_over if side == "plus" else 1 - p_over
                        k1, k2, k3, k4 = st.columns(4)
                        k1.metric(f"{label} — modèle", f"{p_model * 100:.1f} %")
                        k2.metric("Bookmaker", f"{r[f'implicite_{side}'] * 100:.1f} %")
                        k3.metric("Écart (points)", f"{r[f'ecart_{side}_pts']:+.1f}")
                        k4.metric("Valeur théorique", f"{r[f'ev_{side}_pct']:+.1f} %")
                st.warning(
                    "Cet écart vient d'un modèle qui n'a jamais été confronté à de vraies "
                    "cotes de corners ou de cartons. Un grand écart signale le plus souvent "
                    "une information que le modèle n'a pas (compositions, arbitre, enjeu, "
                    "météo), pas une erreur du bookmaker. Ne mise rien sur cette base "
                    "avant un test à terme."
                )

# ----------------------------------------------------------------------
with tab_test:
    st.markdown(
        "**Question posée :** connaître les équipes aide-t-il à prévoir mieux "
        "que la simple moyenne du championnat ? Le modèle est ajusté sur le passé "
        "uniquement, puis testé sur les matchs suivants (3 blocs chronologiques)."
    )

    detail, meta = run_backtest(code, kind, tuple(cfg["lines"]))

    if detail.empty:
        st.warning("Historique insuffisant pour un test fiable sur ce championnat.")
    else:
        summary = cm.summarize_backtest(detail)

        k1, k2, k3 = st.columns(3)
        k1.metric("Matchs testés", meta["n_matchs"])
        k2.metric("Erreur moyenne (modèle)", f"{meta['mae_modele']:.2f}")
        k3.metric("Erreur moyenne (moyenne ligue)", f"{meta['mae_base']:.2f}")

        central = float((detail.groupby("ligne")["y"].mean() - 0.5).abs().idxmin())
        row = summary[summary["ligne"] == central].iloc[0]
        z = row["z"]

        if np.isfinite(z) and z >= 2:
            st.success(
                f"Sur la ligne {central}, le modèle par équipe prévoit nettement mieux "
                f"que la moyenne du championnat (z = {z:.1f}). Connaître les équipes aide."
            )
        elif np.isfinite(z) and z <= -2:
            st.error(
                f"Sur la ligne {central}, le modèle fait moins bien que la simple moyenne "
                f"du championnat (z = {z:.1f}). Ne t'en sers pas ici."
            )
        else:
            st.info(
                f"Sur la ligne {central}, aucune différence nette avec la moyenne du "
                f"championnat (z = {z:.1f}) : l'historique ne permet pas de dire que "
                "connaître les équipes aide ici."
            )

        shown = summary.rename(columns={
            "ligne": "Ligne",
            "matchs": "Matchs",
            "perte_modele": "Perte modèle",
            "perte_moyenne_ligue": "Perte moyenne ligue",
            "gain_moyen": "Gain moyen",
            "erreur_std": "Erreur std",
            "z": "z",
        }).round(4)
        st.dataframe(shown, use_container_width=True, hide_index=True)
        st.caption(
            "Perte = log-loss, plus bas = mieux. Gain moyen = perte de la moyenne ligue "
            "moins perte du modèle : positif = le modèle fait mieux. z au-dessus de 2 = "
            "différence peu probable due au hasard."
        )

        cal = cm.calibration(detail, central)
        if not cal.empty:
            st.markdown(f"**Les probabilités annoncées sont-elles justes ?** (ligne {central}, « plus »)")
            cal = cal.rename(columns={
                "matchs": "Matchs", "annoncee": "Annoncée", "observee": "Observée"
            })
            cal["Annoncée"] = (cal["Annoncée"] * 100).round(1)
            cal["Observée"] = (cal["Observée"] * 100).round(1)
            st.dataframe(cal, use_container_width=True, hide_index=True)
            st.caption("Les deux colonnes (en %) doivent se ressembler. Un gros écart = probabilités à ne pas croire.")

        st.warning(
            "Même si ce test est positif, il ne dit pas qu'on bat les bookmakers : "
            "eux aussi connaissent les équipes, l'arbitre et les compositions. "
            "Pour le savoir, il faut des cotes réelles de corners ou de cartons, "
            "que les fichiers gratuits ne contiennent pas."
        )
