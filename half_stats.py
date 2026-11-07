"""Buts marqués et encaissés par mi-temps, par équipe (moyennes sur les derniers matchs).

Source : colonnes HTHG / HTAG (score à la mi-temps) des fichiers football-data.
Buts de la 2e mi-temps = score final - score à la mi-temps.
"""
import numpy as np
import pandas as pd

import dixon_coles as dc

COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HTHG", "HTAG"]


def load_half(code, n_seasons=2, cache="data"):
    frames = []
    for k, season in enumerate(dc.season_codes(n_seasons)):
        raw = dc._load_csv(code, season, cache, refresh=(k == 0))
        if raw is None or not set(COLS).issubset(raw.columns):
            continue
        d = raw[COLS].copy()
        d["Date"] = pd.to_datetime(d["Date"], dayfirst=True, errors="coerce")
        for c in COLS[3:]:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.dropna()
        frames.append(d)
    if not frames:
        return pd.DataFrame(columns=COLS)
    out = pd.concat(frames, ignore_index=True).drop_duplicates(["Date", "HomeTeam", "AwayTeam"])
    return out.sort_values("Date").reset_index(drop=True)


def averages(df, team, before=None, n=10):
    """Moyennes par match sur les n derniers matchs (avant `before`) :
    buts marqués / encaissés en 1re et en 2e mi-temps. None si pas assez de matchs."""
    if df is None or df.empty:
        return None
    d = df if before is None else df[df["Date"] < pd.Timestamp(before)]
    h = d[d["HomeTeam"] == team]
    a = d[d["AwayTeam"] == team]
    rows = pd.concat([
        pd.DataFrame({"Date": h["Date"], "f1": h["HTHG"], "a1": h["HTAG"],
                      "f2": h["FTHG"] - h["HTHG"], "a2": h["FTAG"] - h["HTAG"]}),
        pd.DataFrame({"Date": a["Date"], "f1": a["HTAG"], "a1": a["HTHG"],
                      "f2": a["FTAG"] - a["HTAG"], "a2": a["FTHG"] - a["HTHG"]}),
    ]).sort_values("Date").tail(n)
    if len(rows) < 3:
        return None
    return {"n": len(rows), "f1": float(rows["f1"].mean()), "a1": float(rows["a1"].mean()),
            "f2": float(rows["f2"].mean()), "a2": float(rows["a2"].mean())}
