"""Biais du marché : que rapporte un pari placé systématiquement sur une
catégorie de cotes, sans aucun modèle ?

Principe
--------
* Chaque match produit 3 paris possibles (1, N, 2), mise de 1 unité.
* Rendement d'un pari = cote x (1 si gagné, 0 sinon) - 1.
* Si le marché est juste (marge du bookmaker retirée proportionnellement),
  le rendement attendu est identique partout : 1/S - 1, où S est la somme
  des probabilités implicites (la "marge" du bookmaker).
* Les catégories positives sur la période ancienne ("découverte") sont
  re-testées sur la période récente ("validation"), avec un seuil statistique
  relevé selon le nombre de catégories candidates (correction de Bonferroni).
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import norm

from dixon_coles import LEAGUES, season_codes, _load_csv

TYPES = ["1", "N", "2"]

EDGES = [1.0, 1.4, 1.8, 2.2, 2.8, 3.5, 5.0, 8.0, 10000.0]
LABELS = [
    "< 1,40",
    "1,40 – 1,79",
    "1,80 – 2,19",
    "2,20 – 2,79",
    "2,80 – 3,49",
    "3,50 – 4,99",
    "5,00 – 7,99",
    "≥ 8,00",
]

# Colonnes des fichiers football-data.co.uk, par source de cotes.
# Pour chaque source, la première série complète d'une ligne est utilisée.
ODDS_SOURCES = {
    "Moyenne du marché (avant match)": [
        ("AvgH", "AvgD", "AvgA"),
        ("B365H", "B365D", "B365A"),
    ],
    "Moyenne de clôture": [("AvgCH", "AvgCD", "AvgCA")],
    "Meilleure cote disponible (Max)": [("MaxH", "MaxD", "MaxA")],
}

MATCH_COLUMNS = ["Date", "FTHG", "FTAG", "o1", "oX", "o2", "League", "Season"]


# --------------------------------------------------------------------------
# Chargement
# --------------------------------------------------------------------------

def _parse_dates(series):
    try:
        return pd.to_datetime(
            series, dayfirst=True, errors="coerce", format="mixed"
        )
    except (TypeError, ValueError):
        return pd.to_datetime(series, dayfirst=True, errors="coerce")


def _extract_odds(raw, triples):
    """Retourne un DataFrame o1/oX/o2 (NaN si cotes absentes ou invalides)."""
    odds = None

    for cols in triples:
        if not all(c in raw.columns for c in cols):
            continue

        cand = raw[list(cols)].apply(pd.to_numeric, errors="coerce")
        cand.columns = ["o1", "oX", "o2"]

        bad = cand.isna().any(axis=1) | (cand <= 1.0).any(axis=1)
        cand.loc[bad, :] = np.nan

        odds = cand if odds is None else odds.combine_first(cand)

    return odds


def load_odds_history(code, source, n_seasons=5, cache="data"):
    """Matchs joués avec cotes 1/N/2 pour un championnat (n saisons)."""
    if source not in ODDS_SOURCES:
        raise ValueError(f"Source de cotes inconnue : {source}")

    triples = ODDS_SOURCES[source]
    frames = []

    for k, season in enumerate(season_codes(n_seasons)):
        raw = _load_csv(code, season, cache, refresh=(k == 0))

        if raw is None or not {"Date", "FTHG", "FTAG"}.issubset(raw.columns):
            continue

        odds = _extract_odds(raw, triples)

        if odds is None:
            continue

        data = pd.DataFrame({
            "Date": _parse_dates(raw["Date"]),
            "FTHG": pd.to_numeric(raw["FTHG"], errors="coerce"),
            "FTAG": pd.to_numeric(raw["FTAG"], errors="coerce"),
        })
        data = pd.concat([data, odds], axis=1)
        data = data.dropna(subset=["Date", "FTHG", "FTAG", "o1", "oX", "o2"])
        data["League"] = code
        data["Season"] = season

        frames.append(data)

    if not frames:
        return pd.DataFrame(columns=MATCH_COLUMNS)

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values("Date").reset_index(drop=True)[MATCH_COLUMNS]


def load_all_odds(source, n_seasons=5, cache="data", leagues=None, workers=8):
    """Tous les championnats : téléchargements en parallèle, puis assemblage."""
    leagues = list(leagues) if leagues is not None else list(LEAGUES)
    seasons = season_codes(n_seasons)

    jobs = [
        (code, season, k == 0)
        for code in leagues
        for k, season in enumerate(seasons)
    ]

    def fetch(job):
        code, season, refresh = job
        try:
            _load_csv(code, season, cache, refresh=refresh)
        except Exception:
            pass

    # Les fichiers manquants sont téléchargés ici ; load_odds_history les
    # relit ensuite depuis le cache disque.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(fetch, jobs))

    frames = []

    for code in leagues:
        part = load_odds_history(code, source, n_seasons, cache)

        if not part.empty:
            frames.append(part)

    if not frames:
        return pd.DataFrame(columns=MATCH_COLUMNS)

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values("Date").reset_index(drop=True)


# --------------------------------------------------------------------------
# Paris et statistiques
# --------------------------------------------------------------------------

def build_bets(matches):
    """Une ligne par pari possible (3 par match)."""
    odds = matches[["o1", "oX", "o2"]].to_numpy(dtype=float)
    overround = (1.0 / odds).sum(axis=1)  # S : somme des proba implicites

    hg = matches["FTHG"].to_numpy()
    ag = matches["FTAG"].to_numpy()
    result = np.where(hg > ag, 0, np.where(hg == ag, 1, 2))

    frames = []

    for k, label in enumerate(TYPES):
        odds_k = odds[:, k]
        win = (result == k).astype(float)

        frames.append(pd.DataFrame({
            "Date": matches["Date"].to_numpy(),
            "League": matches["League"].to_numpy(),
            "Type": label,
            "odds": odds_k,
            "win": win,
            "ret": win * odds_k - 1.0,
            # Rendement attendu si les proba du marché (marge retirée) sont justes
            "fair_ret": 1.0 / overround - 1.0,
            "p_fair": (1.0 / odds_k) / overround,
        }))

    bets = pd.concat(frames, ignore_index=True)
    bets["Tranche"] = pd.cut(
        bets["odds"], bins=EDGES, labels=LABELS, right=False
    ).astype(str)

    return bets[bets["Tranche"].isin(LABELS)].reset_index(drop=True)


def _group_stats(bets, keys):
    grouped = bets.groupby(keys, sort=False)

    out = grouped.agg(
        n=("ret", "size"),
        hit=("win", "mean"),
        p_fair=("p_fair", "mean"),
        expected=("fair_ret", "mean"),
        roi=("ret", "mean"),
        sd=("ret", "std"),
    ).reset_index()

    out["se"] = out["sd"] / np.sqrt(out["n"])
    return out.drop(columns="sd")


def _cell(frame):
    """Statistiques d'un ensemble de paris."""
    n = len(frame)

    if n == 0:
        return {"n": 0, "hit": np.nan, "p_fair": np.nan,
                "expected": np.nan, "roi": np.nan, "se": np.nan}

    ret = frame["ret"].to_numpy()

    return {
        "n": n,
        "hit": float(frame["win"].mean()),
        "p_fair": float(frame["p_fair"].mean()),
        "expected": float(frame["fair_ret"].mean()),
        "roi": float(ret.mean()),
        "se": float(ret.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
    }


def _sort_cells(table):
    t_order = {t: i for i, t in enumerate(TYPES)}
    b_order = {b: i for i, b in enumerate(LABELS)}

    out = table.copy()
    out["_t"] = out["Type"].map(t_order)
    out["_b"] = out["Tranche"].map(b_order)

    return out.sort_values(["_t", "_b"]).drop(columns=["_t", "_b"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Analyse complète
# --------------------------------------------------------------------------

def analyse_bias(matches, validation_fraction=0.4, min_bets=150, alpha=0.05):
    """ROI par type de pari et tranche de cotes, découverte puis validation."""
    if matches is None or len(matches) < 600:
        return {"error": "Pas assez de matchs avec cotes (il en faut au moins 600)."}

    matches = matches.sort_values("Date").reset_index(drop=True)

    cut_index = int(len(matches) * (1.0 - validation_fraction))
    cutoff = matches["Date"].iloc[cut_index]

    bets = build_bets(matches)
    disc = bets[bets["Date"] < cutoff]
    val = bets[bets["Date"] >= cutoff]

    if disc.empty or val.empty:
        return {"error": "Les périodes de découverte et de validation sont vides."}

    # --- par type de pari et tranche de cotes ---
    table = _group_stats(disc, ["Type", "Tranche"]).merge(
        _group_stats(val, ["Type", "Tranche"]),
        on=["Type", "Tranche"],
        how="outer",
        suffixes=("_d", "_v"),
    )
    table = _sort_cells(table)

    n_d = table["n_d"].fillna(0)
    n_v = table["n_v"].fillna(0)

    candidate = (n_d >= min_bets) & (n_v >= min_bets) & (table["roi_d"] > 0)
    n_candidates = int(candidate.sum())

    z_v = table["roi_v"] / table["se_v"]

    if n_candidates > 0:
        # Seuil relevé selon le nombre de catégories candidates.
        z_threshold = float(norm.ppf(1.0 - alpha / n_candidates))
        confirmed = candidate & (table["roi_v"] > 0) & (z_v > z_threshold)
    else:
        z_threshold = float("nan")
        confirmed = candidate & False

    table["z_v"] = z_v
    table["candidate"] = candidate
    table["confirmed"] = confirmed

    # Plus petit gain détectable avec 80 % de chances de le voir : (seuil de
    # significativité + 0,84) x erreur standard d'une catégorie typique de la
    # période de validation. En dessous, un vrai gain passerait inaperçu.
    z_used = 1.96
    if np.isfinite(z_threshold):
        z_used = max(z_used, z_threshold)

    eligible = table["se_v"][(n_v >= min_bets)]
    detectable_roi = (
        float((z_used + 0.84) * np.nanmedian(eligible) * 100.0)
        if len(eligible) > 0
        else float("nan")
    )

    # --- par tranche de cotes seule (graphique) ---
    pooled = _group_stats(disc, ["Tranche"]).merge(
        _group_stats(val, ["Tranche"]),
        on="Tranche",
        how="outer",
        suffixes=("_d", "_v"),
    )
    pooled["_b"] = pooled["Tranche"].map({b: i for i, b in enumerate(LABELS)})
    pooled = pooled.sort_values("_b").drop(columns="_b").reset_index(drop=True)

    # --- par ligue ---
    by_league = bets.groupby("League")
    league_table = pd.DataFrame({
        "Matchs": (by_league.size() / 3.0).round().astype(int),
        "Marge effective (%)": -by_league["fair_ret"].mean() * 100.0,
        "ROI pari aveugle (%)": by_league["ret"].mean() * 100.0,
    }).reset_index()

    return {
        "n_matches": int(len(matches)),
        "n_disc_matches": int((matches["Date"] < cutoff).sum()),
        "n_val_matches": int((matches["Date"] >= cutoff).sum()),
        "date_min": matches["Date"].iloc[0],
        "date_max": matches["Date"].iloc[-1],
        "cutoff": cutoff,
        "table": table,
        "pooled": pooled,
        "league": league_table,
        "overall": {
            "all": _cell(bets),
            "fav": _cell(bets[bets["odds"] < 2.0]),
            "out": _cell(bets[bets["odds"] >= 5.0]),
        },
        "n_candidates": n_candidates,
        "detectable_roi": detectable_roi,
        "z_threshold": z_threshold,
        "alpha": alpha,
        "min_bets": min_bets,
    }


def verdict(report):
    """(niveau, message) : niveau = 'confirmed' ou 'none'."""
    table = report["table"]
    confirmed = table[table["confirmed"]]
    overall = report["overall"]["all"]

    if confirmed.empty:
        if report["n_candidates"] == 0:
            detail = (
                "Aucune catégorie n'était positive sur la période de "
                "découverte."
            )
        else:
            detail = (
                f"{report['n_candidates']} catégorie(s) étaient positives sur "
                "la période de découverte, mais aucune ne l'est restée de "
                "façon statistiquement fiable sur la période de validation."
            )

        power = ""
        if np.isfinite(report.get("detectable_roi", np.nan)):
            power = (
                " Limite de ce test : avec cet échantillon, un gain "
                f"d'environ {report['detectable_roi']:.1f} % par pari ou plus "
                "dans une catégorie serait repéré dans 8 cas sur 10 ; un "
                "avantage plus petit ne peut pas être exclu."
            )

        return "none", (
            "Aucune catégorie de cotes n'est rentable de façon fiable. "
            f"Parier sans modèle sur tous les résultats rapporte en moyenne "
            f"{overall['roi'] * 100:+.1f} % par pari, soit à peu près la marge "
            f"du bookmaker. {detail}{power}"
        )

    lines = []

    for _, r in confirmed.iterrows():
        lines.append(
            f"pari {r['Type']}, cotes {r['Tranche']} : "
            f"ROI validation {r['roi_v'] * 100:+.1f} % "
            f"(IC 95 % ± {1.96 * r['se_v'] * 100:.1f}, {int(r['n_v'])} paris)"
        )

    return "confirmed", (
        "Catégorie(s) positive(s) sur les deux périodes, au-dessus du seuil "
        f"statistique relevé : {' ; '.join(lines)}. À re-vérifier sur une autre "
        "source de cotes et d'autres saisons avant toute mise réelle."
    )


def display_table(report):
    """Tableau lisible (pourcentages) pour l'affichage."""
    t = report["table"]

    def status(row):
        if row["confirmed"]:
            return "Confirmé"
        if row["candidate"]:
            return "Candidat non confirmé"
        return "—"

    out = pd.DataFrame({
        "Pari": t["Type"],
        "Cotes": t["Tranche"],
        "Statut": t.apply(status, axis=1),
        "ROI valid. (%)": t["roi_v"] * 100.0,
        "± IC 95 % (%)": 1.96 * t["se_v"] * 100.0,
        "Paris (valid.)": t["n_v"],
        "ROI découv. (%)": t["roi_d"] * 100.0,
        "Paris (découv.)": t["n_d"],
        "Gagné (%)": t["hit_v"] * 100.0,
        "Proba marché (%)": t["p_fair_v"] * 100.0,
        "ROI si marché juste (%)": t["expected_v"] * 100.0,
    })

    return out


# --------------------------------------------------------------------------
# Graphique
# --------------------------------------------------------------------------

def bias_figure(report):
    """ROI par tranche de cotes : découverte (gris) et validation (bleu)."""
    import plotly.graph_objects as go

    p = report["pooled"]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Découverte (période ancienne)",
        x=p["Tranche"],
        y=p["roi_d"] * 100.0,
        error_y=dict(type="data", array=(1.96 * p["se_d"] * 100.0).fillna(0)),
        marker_color="#8d8c86",
        hovertemplate="%{x}<br>ROI %{y:+.1f} %<extra>Découverte</extra>",
    ))

    fig.add_trace(go.Bar(
        name="Validation (période récente)",
        x=p["Tranche"],
        y=p["roi_v"] * 100.0,
        error_y=dict(type="data", array=(1.96 * p["se_v"] * 100.0).fillna(0)),
        marker_color="#2a78d6",
        hovertemplate="%{x}<br>ROI %{y:+.1f} %<extra>Validation</extra>",
    ))

    fig.add_hline(y=0, line_width=1, line_color="#52514e")

    expected = report["overall"]["all"]["expected"]
    if np.isfinite(expected):
        fig.add_hline(
            y=expected * 100.0,
            line_width=1,
            line_dash="dash",
            line_color="#eb6834",
            annotation_text="Marché juste : perte = marge",
            annotation_position="bottom right",
        )

    fig.update_layout(
        barmode="group",
        bargap=0.25,
        yaxis_title="ROI par pari (%)",
        xaxis_title="Cote du pari",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)

    return fig
