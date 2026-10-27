import os, time
import numpy as np
import pandas as pd
import requests

URL = "https://www.football-data.co.uk/fixtures.csv"
REQUIRED = ["Div", "HomeTeam", "AwayTeam"]


def load_fixtures(cache="data"):
    os.makedirs(cache, exist_ok=True)
    path = f"{cache}/fixtures.csv"
    if not os.path.exists(path) or time.time() - os.path.getmtime(path) > 3600:
        try:
            r = requests.get(URL, timeout=30)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)
        except Exception as e:
            print(f"fixtures.csv: telechargement impossible ({e})")
    if not os.path.exists(path):
        return pd.DataFrame()
    d = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            d = pd.read_csv(path, encoding=enc)
            break
        except Exception:
            continue
    if d is None or d.empty:
        return pd.DataFrame()
    # nettoie les noms de colonnes (BOM, espaces)
    d.columns = [str(c).replace("\ufeff", "").replace("ï»¿", "").strip()
                 for c in d.columns]
    if any(c not in d.columns for c in REQUIRED):
        return pd.DataFrame()
    d = d.dropna(subset=REQUIRED)
    if "Date" in d.columns:
        d["Date"] = pd.to_datetime(d["Date"], dayfirst=True, errors="coerce")
    else:
        d["Date"] = pd.NaT
    for c in ["B365H", "B365D", "B365A", "AvgH", "AvgD", "AvgA"]:
        if c not in d.columns:
            d[c] = np.nan
    return d.sort_values("Date").reset_index(drop=True)


def get_odds(row):
    """Cotes moyennes du marche, sinon Bet365. None si indisponibles."""
    for cols in (("AvgH", "AvgD", "AvgA"), ("B365H", "B365D", "B365A")):
        v = [row.get(c, np.nan) for c in cols]
        if not any(pd.isna(x) for x in v):
            return tuple(float(x) for x in v)
    return None
