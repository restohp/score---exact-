import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from dixon_coles import LEAGUES, load_league, DixonColes
from fixtures import load_fixtures, get_odds
import market_bias as mb

st.set_page_config(page_title="Score Exact Pro", layout="wide")
st.title("⚽ Score Exact Pro")
st.caption(
    "Outil d'analyse statistique — les scores et les Value Bets "
    "ne sont jamais garantis."
)

WEIGHTS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


@st.cache_data(ttl=43200, show_spinner="Chargement de l'historique...")
def cached_data(code):
    return load_league(code, n_seasons=4)


@st.cache_resource(show_spinner="Entraînement du modèle...")
def cached_model(code, latest_date):
    df = cached_data(code)
    return DixonColes().fit(df, ref_date=pd.Timestamp(latest_date))


@st.cache_data(ttl=3600, show_spinner=False)
def cached_fixtures(code):
    """Matchs pas encore joués et absents de l'historique (évite la fuite de données)."""
    fx = load_fixtures()

    if fx.empty:
        return fx

    fx = fx[fx["Div"] == code].copy()

    today = pd.Timestamp.today().normalize()
    fx = fx[fx["Date"].isna() | (fx["Date"] >= today)]

    df = cached_data(code)
    played = set(zip(df.HomeTeam, df.AwayTeam, df.Date.dt.date))

    keep = []

    for h, a, d in zip(fx.HomeTeam, fx.AwayTeam, fx.Date):
        dd = d.date() if pd.notna(d) else None
        keep.append((h, a, dd) not in played)

    return fx.loc[np.array(keep, dtype=bool)].reset_index(drop=True)


def fmt_date(value):
    return (
        value.strftime("%d/%m/%Y")
        if pd.notna(value)
        else "Date inconnue"
    )


def blend_probs(p_model, odds, weight_model):
    """Mélange modèle/marché après normalisation du marché."""
    if odds is None:
        return p_model.copy(), None

    odds = np.asarray(odds, dtype=float)

    if np.any(~np.isfinite(odds)) or np.any(odds <= 1):
        return p_model.copy(), None

    q = 1.0 / odds
    q /= q.sum()

    blended = weight_model * p_model + (1 - weight_model) * q
    return blended, q


def adjust_matrix(matrix, target_1x2):
    """Recale les probabilités pour correspondre au 1/N/2 cible."""
    original = DixonColes.probs_1x2(matrix)

    factors = target_1x2 / np.clip(original, 1e-12, None)

    out = matrix.copy()

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            group = 0 if i > j else (1 if i == j else 2)
            out[i, j] = matrix[i, j] * factors[group]

    return out / out.sum()


def analyse(model, home, away, odds=None, model_weight=1.0, top=5):
    matrix = model.score_matrix(home, away)
    lh, la = model.rates(home, away)

    p_model = DixonColes.probs_1x2(matrix)
    p_blend, q = blend_probs(p_model, odds, model_weight)

    adjusted = adjust_matrix(matrix, p_blend)

    scores = sorted(
        (
            (i, j, adjusted[i, j])
            for i in range(adjusted.shape[0])
            for j in range(adjusted.shape[1])
        ),
        key=lambda x: x[2],
        reverse=True,
    )[:top]

    over25 = sum(
        adjusted[i, j]
        for i in range(adjusted.shape[0])
        for j in range(adjusted.shape[1])
        if i + j > 2
    )

    btts = (
        1
        - adjusted[0, :].sum()
        - adjusted[:, 0].sum()
        + adjusted[0, 0]
    )

    return {
        "matrix": adjusted,
        "lh": lh,
        "la": la,
        "p_model": p_model,
        "q": q,
        "p_blend": p_blend,
        "top": [
            (f"{i}-{j}", p * 100)
            for i, j, p in scores
        ],
        "over25": over25 * 100,
        "btts": btts * 100,
    }


def brier(probabilities, actual):
    """Score de Brier multiclasses pour les résultats 1/N/2."""
    onehot = np.eye(3)[np.asarray(actual, dtype=int)]

    return np.sum(
        (np.asarray(probabilities) - onehot) ** 2,
        axis=1,
    )


def chronological_backtest(
    df,
    n_test=300,
    calibration_fraction=0.4,
    refit_days=14,
):
    """
    Backtest chronologique en deux blocs.

    1. Calibration : choix du poids modèle/marché.
    2. Test final : évaluation sur une période indépendante.

    Le modèle est réentraîné tous les `refit_days` jours, uniquement
    sur les matchs antérieurs au début de chaque période.
    """

    d = df.sort_values("Date").reset_index(drop=True)

    n_test = min(int(n_test), len(d) // 2)

    if n_test < 60:
        return None

    test = d.iloc[len(d) - n_test:].copy().reset_index(drop=True)

    split = max(
        20,
        min(
            len(test) - 20,
            int(len(test) * calibration_fraction),
        ),
    )

    calibration = test.iloc[:split]
    final = test.iloc[split:]

    if len(final) < 20:
        return None

    def make_predictions(rows):
        records = []

        if rows.empty:
            return records

        origin = rows["Date"].min()
        bucket = (
            (rows["Date"] - origin).dt.days // refit_days
        ).to_numpy()

        for b in np.unique(bucket):
            part = rows[bucket == b]
            start_date = part["Date"].min()

            train = d[d["Date"] < start_date]

            if len(train) < 80:
                continue

            try:
                model = DixonColes().fit(
                    train,
                    ref_date=start_date,
                )
            except (
                ValueError,
                RuntimeError,
                FloatingPointError,
            ):
                continue

            for _, row in part.iterrows():
                odds = get_odds(row)

                if odds is None:
                    continue

                if (
                    row["HomeTeam"] not in model.idx
                    or row["AwayTeam"] not in model.idx
                ):
                    continue

                try:
                    matrix = model.score_matrix(
                        row["HomeTeam"],
                        row["AwayTeam"],
                    )

                    p_model = DixonColes.probs_1x2(matrix)

                    q = 1 / np.asarray(odds, dtype=float)
                    q /= q.sum()

                    actual = (
                        0
                        if row["FTHG"] > row["FTAG"]
                        else 1
                        if row["FTHG"] == row["FTAG"]
                        else 2
                    )

                    records.append((p_model, q, actual))

                except (
                    ValueError,
                    KeyError,
                    RuntimeError,
                    FloatingPointError,
                ):
                    continue

        return records

    cal = make_predictions(calibration)
    holdout = make_predictions(final)

    if len(cal) < 20 or len(holdout) < 20:
        return {
            "error": (
                "Pas assez de matchs avec des cotes historiques "
                "valides dans les deux blocs."
            )
        }

    # Le poids est choisi uniquement sur la calibration.
    choices = []

    for weight in WEIGHTS:
        losses = []

        for p, q, actual in cal:
            prediction = weight * p + (1 - weight) * q

            losses.append(
                brier([prediction], [actual])[0]
            )

        choices.append((float(np.mean(losses)), weight))

    calibration_brier, best_weight = min(choices)

    def summarize(records, weight):
        market_losses = []
        model_losses = []
        blend_losses = []

        for p, q, actual in records:
            market_losses.append(
                brier([q], [actual])[0]
            )

            model_losses.append(
                brier([p], [actual])[0]
            )

            blend = weight * p + (1 - weight) * q

            blend_losses.append(
                brier([blend], [actual])[0]
            )

        diffs = (
            np.asarray(blend_losses)
            - np.asarray(market_losses)
        )

        se = (
            float(
                diffs.std(ddof=1) / np.sqrt(len(diffs))
            )
            if len(diffs) > 1
            else float("nan")
        )

        return {
            "n": len(records),
            "market": float(np.mean(market_losses)),
            "model": float(np.mean(model_losses)),
            "blend": float(np.mean(blend_losses)),
            "diff": float(diffs.mean()),
            "se": se,
            "significant": bool(
                np.isfinite(se)
                and diffs.mean() < -2 * se
            ),
        }

    return {
        "calibration_n": len(cal),
        "final": summarize(holdout, best_weight),
        "weight": best_weight,
        "calibration_brier": calibration_brier,
    }


@st.cache_data(ttl=43200, show_spinner=False)
def run_backtest(code, n_test):
    return chronological_backtest(cached_data(code), n_test=n_test)


@st.cache_data(ttl=86400, show_spinner=False)
def run_market_bias(source):
    matches = mb.load_all_odds(source, n_seasons=5)
    return mb.analyse_bias(matches)


# --------------------------------------------------
# Configuration du championnat
# --------------------------------------------------

code = st.sidebar.selectbox(
    "Championnat",
    list(LEAGUES),
    format_func=lambda c: f"{LEAGUES[c]} ({c})",
    index=list(LEAGUES).index("F1"),
)

try:
    df = cached_data(code)
except Exception as exc:
    st.error(f"Impossible de charger les données : {exc}")
    st.stop()

latest = df["Date"].max()

try:
    model = cached_model(code, str(latest))
except Exception as exc:
    st.error(f"Impossible d'entraîner le modèle : {exc}")
    st.stop()


# --------------------------------------------------
# Chargement des matchs
# --------------------------------------------------

fixtures = cached_fixtures(code)

recent_df = df[
    df.Date > df.Date.max() - pd.Timedelta(days=400)
]

recent = sorted(
    set(recent_df.HomeTeam) | set(recent_df.AwayTeam)
)

mode = "Manuel"

if not fixtures.empty:
    mode = st.sidebar.radio(
        "Source du match",
        ["Matchs à venir", "Manuel"],
    )

odds_known = False

if mode == "Matchs à venir":
    labels = [
        f"{fmt_date(r.Date)} — {r.HomeTeam} - {r.AwayTeam}"
        for r in fixtures.itertuples()
    ]

    selected_index = st.sidebar.selectbox(
        "Match",
        range(len(fixtures)),
        format_func=lambda k: labels[k],
    )

    selected_row = fixtures.iloc[selected_index]

    home = selected_row.HomeTeam
    away = selected_row.AwayTeam

    odds = get_odds(selected_row)

    if odds:
        odds_known = True
    else:
        odds = (2.00, 3.40, 3.60)

else:
    if len(recent) < 2:
        st.error("Pas assez d'équipes dans les données.")
        st.stop()

    home = st.sidebar.selectbox(
        "Domicile",
        recent,
        index=0,
    )

    away_options = [
        team for team in recent if team != home
    ]

    away = st.sidebar.selectbox(
        "Extérieur",
        away_options,
        index=0,
    )

    odds = (2.00, 3.40, 3.60)


# --------------------------------------------------
# Cotes du bookmaker
# --------------------------------------------------

st.sidebar.markdown("**Cotes du bookmaker**")

o1 = st.sidebar.number_input(
    "Cote 1",
    min_value=1.01,
    max_value=100.0,
    value=float(odds[0]),
    step=0.05,
)

oX = st.sidebar.number_input(
    "Cote N",
    min_value=1.01,
    max_value=100.0,
    value=float(odds[1]),
    step=0.05,
)

o2 = st.sidebar.number_input(
    "Cote 2",
    min_value=1.01,
    max_value=100.0,
    value=float(odds[2]),
    step=0.05,
)

use_odds = odds_known or st.sidebar.checkbox(
    "J'ai saisi les vraies cotes",
    value=False,
    help=(
        "Coche uniquement si les trois cotes saisies "
        "correspondent réellement au bookmaker."
    ),
)

model_weight = st.sidebar.slider(
    "Poids du modèle (0 = marché, 1 = modèle)",
    0.0,
    1.0,
    0.0,
    0.1,
)


# --------------------------------------------------
# Onglets
# --------------------------------------------------

tab0, tab1, tab2, tab3 = st.tabs(
    [
        "Matchs à venir",
        "Prédiction",
        "Backtest indépendant",
        "Biais du marché",
    ]
)


# --------------------------------------------------
# Onglet 1 : matchs à venir
# --------------------------------------------------

with tab0:
    if fixtures.empty:
        st.info(
            "Aucun match à venir pour cette ligue : le fichier ne "
            "contient que des matchs déjà joués, ou n'est pas encore "
            "mis à jour. Utilise le mode Manuel dans la barre latérale."
        )

    else:
        rows = []

        for _, f in fixtures.iterrows():
            h, a = f.HomeTeam, f.AwayTeam

            if h not in model.idx or a not in model.idx:
                continue

            try:
                o = get_odds(f)

                result = analyse(
                    model,
                    h,
                    a,
                    o,
                    model_weight if o else 1.0,
                    top=3,
                )

                ev = (
                    (result["p_blend"] * np.asarray(o)).tolist()
                    if o
                    else [np.nan] * 3
                )

                best = int(np.argmax(ev)) if o else None

                rows.append({
                    "Date": fmt_date(f.Date),
                    "Match": f"{h} - {a}",
                    "Buts attendus": (
                        f"{result['lh']:.2f} - "
                        f"{result['la']:.2f}"
                    ),
                    "Score probable 1": (
                        f"{result['top'][0][0]} "
                        f"({result['top'][0][1]:.1f}%)"
                    ),
                    "Score probable 2": (
                        f"{result['top'][1][0]} "
                        f"({result['top'][1][1]:.1f}%)"
                    ),
                    "Score probable 3": (
                        f"{result['top'][2][0]} "
                        f"({result['top'][2][1]:.1f}%)"
                    ),
                    "P(1)/P(N)/P(2)": "/".join(
                        f"{p * 100:.0f}"
                        for p in result["p_blend"]
                    ),
                    "EV max": (
                        round(ev[best], 3)
                        if best is not None
                        else np.nan
                    ),
                    "Issue EV": (
                        ["1", "N", "2"][best]
                        if best is not None
                        else "Cotes absentes"
                    ),
                })

            except (KeyError, ValueError, RuntimeError):
                continue

        if rows:
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )

        st.caption(
            "Une EV supérieure à 1 est une valeur théorique positive "
            "selon le modèle, pas une garantie. "
            "Les résultats doivent être validés hors échantillon."
        )


# --------------------------------------------------
# Onglet 2 : prédiction
# --------------------------------------------------

with tab1:
    if home == away:
        st.warning("Choisis deux équipes différentes.")

    elif home not in model.idx or away not in model.idx:
        st.warning(
            "Une équipe n'est pas dans l'historique du modèle."
        )

    else:
        odds_arg = (
            (o1, oX, o2)
            if use_odds
            else None
        )

        result = analyse(
            model,
            home,
            away,
            odds_arg,
            model_weight if odds_arg else 1.0,
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Buts attendus",
            f"{result['lh']:.2f} – {result['la']:.2f}",
        )

        c2.metric(
            "Plus de 2,5 buts",
            f"{result['over25']:.1f}%",
        )

        c3.metric(
            "Les deux équipes marquent",
            f"{result['btts']:.1f}%",
        )

        st.subheader(
            f"{home} – {away} : scores exacts les plus probables"
        )

        st.dataframe(
            pd.DataFrame(
                result["top"],
                columns=["Score", "Probabilité (%)"],
            ),
            use_container_width=True,
            hide_index=True,
        )

        grid = result["matrix"][:7, :7] * 100

        fig = px.imshow(
            grid,
            text_auto=".1f",
            aspect="auto",
            color_continuous_scale="Blues",
            labels={
                "x": f"Buts de {away}",
                "y": f"Buts de {home}",
                "color": "Probabilité (%)",
            },
        )

        fig.update_xaxes(fixedrange=True)
        fig.update_yaxes(fixedrange=True)

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False},
        )

        if use_odds:
            odds_array = np.array([o1, oX, o2])
            ev = result["p_blend"] * odds_array

            table = pd.DataFrame({
                "Issue": ["1", "N", "2"],
                "Probabilité modèle (%)": (
                    result["p_model"] * 100
                ),
                "Probabilité marché normalisée (%)": (
                    result["q"] * 100
                ),
                "Probabilité mélange (%)": (
                    result["p_blend"] * 100
                ),
                "EV théorique": ev,
            })

            st.subheader("Comparaison au marché")

            st.dataframe(
                table.style.format({
                    "Probabilité modèle (%)": "{:.1f}",
                    "Probabilité marché normalisée (%)": "{:.1f}",
                    "Probabilité mélange (%)": "{:.1f}",
                    "EV théorique": "{:.3f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                f"Marge implicite brute : "
                f"{((1 / odds_array).sum() - 1) * 100:.1f}%. "
                "La normalisation retire proportionnellement la marge "
                "du marché, sans garantir une correction parfaite."
            )

        else:
            st.info(
                "Entre les vraies cotes puis coche la case dans "
                "la barre latérale pour activer la comparaison."
            )


# --------------------------------------------------
# Onglet 3 : backtest indépendant
# --------------------------------------------------

with tab2:
    st.write(
        "Le premier bloc chronologique choisit le poids du mélange. "
        "Le second bloc, indépendant, évalue ce choix. "
        "Le modèle est réentraîné tous les 14 jours."
    )

    n_test = st.selectbox(
        "Taille de la période de test "
        "(matchs les plus récents)",
        [120, 180, 300],
        index=2,
    )

    if st.button("Lancer le backtest indépendant"):
        with st.spinner("Backtest en cours (1 à 3 min)..."):
            try:
                st.session_state["bt"] = (
                    code,
                    n_test,
                    run_backtest(code, n_test),
                )
            except Exception as exc:
                st.session_state["bt"] = (
                    code,
                    n_test,
                    {"error": f"Le backtest a échoué : {exc}"},
                )

    saved = st.session_state.get("bt")

    if saved and saved[0] == code and saved[1] == n_test:
        report = saved[2]

        if report is None:
            st.warning(
                "Pas assez de données historiques pour ce test."
            )

        elif "error" in report:
            st.warning(report["error"])

        else:
            final = report["final"]

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Matchs calibration",
                report["calibration_n"],
            )

            c2.metric(
                "Matchs test final",
                final["n"],
            )

            c3.metric(
                "Poids retenu",
                f"{report['weight']:.1f}",
            )

            st.dataframe(
                pd.DataFrame([
                    {
                        "Méthode": "Marché normalisé",
                        "Brier final": final["market"],
                    },
                    {
                        "Méthode": "Modèle seul",
                        "Brier final": final["model"],
                    },
                    {
                        "Méthode": (
                            f"Mélange "
                            f"(poids {report['weight']:.1f})"
                        ),
                        "Brier final": final["blend"],
                    },
                ]).style.format({
                    "Brier final": "{:.4f}"
                }),
                use_container_width=True,
                hide_index=True,
            )

            st.write(
                f"Écart mélange − marché : "
                f"**{final['diff']:+.4f} ± {final['se']:.4f}** "
                "(erreur standard approximative)."
            )

            if final["significant"]:
                st.success(
                    "Le mélange améliore le Brier du marché de plus "
                    "de deux erreurs standards sur ce test. "
                    "Répète sur d'autres ligues et périodes avant "
                    "de conclure."
                )
            else:
                st.warning(
                    "Aucune preuve statistique suffisante que le "
                    "mélange bat le marché sur ce test."
                )

            st.caption(
                "Ce test est plus rigoureux que choisir et évaluer "
                "le poids sur les mêmes matchs. Il reste sensible "
                "au championnat, aux cotes disponibles, à la qualité "
                "des données et aux changements de saison."
            )


# --------------------------------------------------
# Onglet 4 : biais du marché
# --------------------------------------------------

with tab3:
    st.write(
        "Question : parier systématiquement sur une catégorie de cotes "
        "(favoris, nuls, outsiders), sans aucun modèle, rapporte-t-il plus "
        "que la marge du bookmaker ne coûte ? Le test utilise tous les "
        "championnats et 5 saisons. Les catégories positives sur la période "
        "ancienne sont re-testées sur la période récente, avec un seuil "
        "statistique relevé pour limiter les coïncidences."
    )

    source = st.selectbox(
        "Cotes utilisées",
        list(mb.ODDS_SOURCES),
        index=0,
    )

    if source.startswith("Meilleure"):
        st.caption(
            "Les meilleures cotes de plusieurs bookmakers sont plus "
            "favorables que celles d'un seul : un ROI meilleur est normal. "
            "Elles ne sont pas toujours obtenables en pratique (limites de "
            "mise, cotes périmées ou erronées)."
        )

    if st.button("Lancer l'analyse du biais du marché"):
        with st.spinner(
            "Téléchargement et calcul (1 à 3 min la première fois)..."
        ):
            try:
                st.session_state["mb"] = (source, run_market_bias(source))
            except Exception as exc:
                st.session_state["mb"] = (
                    source,
                    {"error": f"L'analyse a échoué : {exc}"},
                )

    saved_mb = st.session_state.get("mb")

    if saved_mb and saved_mb[0] == source:
        bias = saved_mb[1]

        if "error" in bias:
            st.warning(bias["error"])

        else:
            overall = bias["overall"]

            c1, c2, c3 = st.columns(3)

            c1.metric("Matchs analysés", f"{bias['n_matches']:,}".replace(",", " "))
            c2.metric(
                "Période",
                f"{bias['date_min']:%m/%Y} – {bias['date_max']:%m/%Y}",
            )
            c3.metric(
                "Séparation découverte / validation",
                f"{bias['cutoff']:%d/%m/%Y}",
            )

            c4, c5, c6 = st.columns(3)

            c4.metric(
                "Tous les paris (ROI)",
                f"{overall['all']['roi'] * 100:+.1f} %",
            )
            c5.metric(
                "Favoris, cote < 2,00",
                f"{overall['fav']['roi'] * 100:+.1f} %",
            )
            c6.metric(
                "Outsiders, cote ≥ 5,00",
                f"{overall['out']['roi'] * 100:+.1f} %",
            )

            level, message = mb.verdict(bias)

            if level == "confirmed":
                st.success(message)
            else:
                st.info(message)

            st.subheader("ROI par tranche de cotes")

            st.plotly_chart(
                mb.bias_figure(bias),
                use_container_width=True,
                config={"displayModeBar": False},
            )

            st.caption(
                "Les barres d'erreur sont des intervalles de confiance à "
                "95 %. La ligne pointillée est la perte attendue si le "
                "marché est juste (la marge du bookmaker)."
            )

            st.subheader("Détail par type de pari")

            detail = mb.display_table(bias)

            st.dataframe(
                detail.style.format(
                    {
                        "ROI valid. (%)": "{:+.1f}",
                        "± IC 95 % (%)": "{:.1f}",
                        "Paris (valid.)": "{:.0f}",
                        "ROI découv. (%)": "{:+.1f}",
                        "Paris (découv.)": "{:.0f}",
                        "Gagné (%)": "{:.1f}",
                        "Proba marché (%)": "{:.1f}",
                        "ROI si marché juste (%)": "{:+.1f}",
                    },
                    na_rep="—",
                ),
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("Par championnat")

            league = bias["league"].copy()
            league["League"] = league["League"].map(
                lambda c: LEAGUES.get(c, c)
            )
            league = league.rename(columns={"League": "Championnat"})

            st.dataframe(
                league.style.format(
                    {
                        "Marge effective (%)": "{:.1f}",
                        "ROI pari aveugle (%)": "{:+.1f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "Lecture : un « Candidat non confirmé » était positif sur "
                "la période ancienne mais pas de façon fiable sur la période "
                "récente, ce qui est le résultat attendu d'un simple hasard. "
                "Les cotes viennent de football-data.co.uk, avant le match "
                "(ou à la clôture selon ton choix) ; elles ne contiennent "
                "ni frais ni limites de mise. Le test ne couvre que le "
                "1X2, pas le score exact."
            )
