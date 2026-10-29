"""Calendrier des matchs : prévisions du modèle, cotes du marché, résultats.

Règle anti-fuite : pour une période donnée, le modèle est ajusté
uniquement sur les matchs STRICTEMENT antérieurs au début de la période
(ou sur tout l'historique si la période est dans le futur). Les matchs
déjà joués dans la période sont donc prédits "comme avant le coup d'envoi".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import dixon_coles as dc
from dixon_coles import DixonColes
import fixtures as fx

JOURS = ["lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim."]
MOIS = ["janv.", "févr.", "mars", "avr.", "mai", "juin",
        "juil.", "août", "sept.", "oct.", "nov.", "déc."]
ISSUES = ["1", "N", "2"]
MIN_TRAIN = 200


def fr_date(ts):
    ts = pd.Timestamp(ts)
    return f"{JOURS[ts.weekday()]} {ts.day} {MOIS[ts.month - 1]} {ts.year}"


def cutoff_for(df, start):
    """Date avant laquelle le modèle a le droit de s'entraîner."""
    last = df["Date"].max()
    return min(pd.Timestamp(start), last + pd.Timedelta(days=1))


def fit_before(df, cutoff):
    train = df[df["Date"] < cutoff]
    if len(train) < MIN_TRAIN:
        raise ValueError(
            "Pas assez de matchs avant cette période pour entraîner le modèle."
        )
    return DixonColes().fit(train, ref_date=cutoff - pd.Timedelta(days=1))


def _analyse(model, home, away):
    matrix = model.score_matrix(home, away)
    p = DixonColes.probs_1x2(matrix)
    i, j = np.unravel_index(np.argmax(matrix), matrix.shape)
    return p, f"{i}-{j}", float(matrix[i, j])


def _row(code, name, status, date, home, away, model, odds, hg=None, ag=None):
    p, score, p_score = _analyse(model, home, away)
    market = dc.market_probabilities(odds) if odds else None
    result = None
    if hg is not None and ag is not None:
        result = dc.outcome_1x2(int(hg), int(ag))
    return {
        "date": pd.Timestamp(date).normalize(),
        "code": code,
        "ligue": name,
        "statut": status,
        "home": home,
        "away": away,
        "hg": None if hg is None else int(hg),
        "ag": None if ag is None else int(ag),
        "result": result,
        "p_model": p,
        "p_market": market,
        "odds": odds,
        "score": score,
        "p_score": p_score,
    }


def collect(code, df, fixtures_df, model, start, end):
    """Matchs de la période pour un championnat : (lignes, nb non analysés)."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    name = dc.LEAGUES[code]
    rows, skipped = [], 0

    # 1) matchs déjà joués dans la période
    played = df[(df["Date"] >= start) & (df["Date"] <= end)]
    for _, r in played.iterrows():
        if r["HomeTeam"] not in model.idx or r["AwayTeam"] not in model.idx:
            skipped += 1
            continue
        rows.append(_row(code, name, "Joué", r["Date"], r["HomeTeam"],
                         r["AwayTeam"], model, dc.valid_odds(r),
                         r["FTHG"], r["FTAG"]))

    # 2) matchs à venir (fixtures.csv), hors matchs déjà dans l'historique
    if fixtures_df is not None and not fixtures_df.empty:
        f = fixtures_df[(fixtures_df["Div"] == code)
                        & fixtures_df["Date"].notna()
                        & (fixtures_df["Date"] >= start)
                        & (fixtures_df["Date"] <= end)]
        seen = set(zip(df["HomeTeam"], df["AwayTeam"], df["Date"].dt.date))
        for _, r in f.iterrows():
            if (r["HomeTeam"], r["AwayTeam"], r["Date"].date()) in seen:
                continue
            if r["HomeTeam"] not in model.idx or r["AwayTeam"] not in model.idx:
                skipped += 1
                continue
            rows.append(_row(code, name, "À venir", r["Date"], r["HomeTeam"],
                             r["AwayTeam"], model, fx.get_odds(r)))
    return rows, skipped


def brier(p, result):
    target = np.zeros(3)
    target[result] = 1.0
    return float(((np.asarray(p) - target) ** 2).sum())


def summarize_played(rows):
    """Bilan sur les matchs joués : modèle vs marché (mêmes matchs)."""
    both = [r for r in rows
            if r["result"] is not None and r["p_market"] is not None]
    if not both:
        return None
    acc_m = np.mean([int(np.argmax(r["p_model"]) == r["result"]) for r in both])
    acc_k = np.mean([int(np.argmax(r["p_market"]) == r["result"]) for r in both])
    bm = np.array([brier(r["p_model"], r["result"]) for r in both])
    bk = np.array([brier(r["p_market"], r["result"]) for r in both])
    diff = bm - bk  # > 0 : le modèle fait moins bien que le marché
    se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else np.nan
    return {
        "n": len(both),
        "acc_model": float(acc_m),
        "acc_market": float(acc_k),
        "brier_model": float(bm.mean()),
        "brier_market": float(bk.mean()),
        "diff": float(diff.mean()),
        "se": float(se),
    }


def to_table(rows):
    """Tableau lisible (un DataFrame par jour est construit par la page)."""
    out = []
    for r in rows:
        pick = int(np.argmax(r["p_model"]))
        if r["result"] is None:
            mark, res = "", "—"
        else:
            mark = "✔" if pick == r["result"] else "✘"
            res = f"{r['hg']}-{r['ag']}"
        out.append({
            "date": r["date"],
            "Match": f"{r['home']} - {r['away']}",
            "Résultat": res,
            "Pronostic": ISSUES[pick],
            "Juste ?": mark,
            "Modèle 1/N/2 (%)": "/".join(f"{x * 100:.0f}" for x in r["p_model"]),
            "Marché 1/N/2 (%)": ("/".join(f"{x * 100:.0f}" for x in r["p_market"])
                                 if r["p_market"] is not None else "—"),
            "Cotes": (" / ".join(f"{o:.2f}" for o in r["odds"])
                      if r["odds"] else "—"),
            "Score probable": f"{r['score']} ({r['p_score'] * 100:.0f} %)",
            "Ligue": r["ligue"],
        })
    return pd.DataFrame(out)
