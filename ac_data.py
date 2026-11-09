"""Auto-correction (1/3) : données, prédictions « dans le passé », outils de mi-temps."""
import numpy as np
import pandas as pd
from scipy.stats import poisson

import dixon_coles as dc
import forme_core as fc
from dixon_coles import DixonColes

HALF_SHARE = 0.45    # part des buts en 1re mi-temps (moyenne des grands championnats)
HALF_PSEUDO = 20     # buts fictifs pour la part de mi-temps d'une équipe (comme sur la carte)


# ----------------------------------------------------------------------
# 0. Données : résultats finaux + score à la mi-temps
# ----------------------------------------------------------------------
def load_data(code, n_seasons=5, cache="data"):
    """Résultats (fc.load_stats) + colonnes HTHG / HTAG (score à la mi-temps) si disponibles."""
    df = fc.load_stats(code, n_seasons=n_seasons, cache=cache)
    frames = []
    for season in dc.season_codes(n_seasons):
        try:
            raw = dc._load_csv(code, season, cache, refresh=False)
        except Exception:
            raw = None
        need = {"Date", "HomeTeam", "AwayTeam", "HTHG", "HTAG"}
        if raw is None or not need.issubset(raw.columns):
            continue
        d = raw[["Date", "HomeTeam", "AwayTeam", "HTHG", "HTAG"]].copy()
        d["Date"] = pd.to_datetime(d["Date"], dayfirst=True, errors="coerce")
        d["HTHG"] = pd.to_numeric(d["HTHG"], errors="coerce")
        d["HTAG"] = pd.to_numeric(d["HTAG"], errors="coerce")
        frames.append(d.dropna())
    if frames:
        h = pd.concat(frames, ignore_index=True).drop_duplicates(["Date", "HomeTeam", "AwayTeam"])
        df = df.merge(h, on=["Date", "HomeTeam", "AwayTeam"], how="left")
    else:
        df["HTHG"] = np.nan
        df["HTAG"] = np.nan
    return df


# ----------------------------------------------------------------------
# 1. Matrice de scores à partir de deux taux de buts
# ----------------------------------------------------------------------
def matrix_from_rates(lh, la, rho, max_goals=8):
    """Même matrice que DixonColes.score_matrix, mais à partir de taux de buts donnés."""
    goals = np.arange(max_goals + 1)
    m = np.outer(poisson.pmf(goals, lh), poisson.pmf(goals, la))
    m[0, 0] *= max(1 - lh * la * rho, 1e-8)
    m[0, 1] *= max(1 + lh * rho, 1e-8)
    m[1, 0] *= max(1 + la * rho, 1e-8)
    m[1, 1] *= max(1 - rho, 1e-8)
    return m / m.sum()


def _probs(m):
    n = m.shape[0]
    tg = np.add.outer(np.arange(n), np.arange(n))
    p = DixonColes.probs_1x2(m)
    return (p[0], p[1], p[2], float(m[tg >= 2].sum()),
            float(1 - m[0, :].sum() - m[:, 0].sum() + m[0, 0]))


def half_scores(i, j, share):
    """Score probable de chaque mi-temps sachant le score final probable i-j
    (identique à cartes_core.half_scores)."""
    h1 = min(int(np.floor((i + 1) * share)), i)
    a1 = min(int(np.floor((j + 1) * share)), j)
    return (h1, a1), (i - h1, j - a1)


def _team_logs(df):
    """Par équipe : dates, buts en 1re mi-temps (total du match), buts en 2e mi-temps."""
    if "HTHG" not in df.columns:
        return {}
    d = df.dropna(subset=["HTHG", "HTAG"]).sort_values("Date")
    g1 = (d["HTHG"] + d["HTAG"]).to_numpy(float)
    g2 = (d["FTHG"] + d["FTAG"]).to_numpy(float) - g1
    dates = d["Date"].to_numpy()
    out = {}
    for team in pd.unique(pd.concat([d["HomeTeam"], d["AwayTeam"]])):
        mk = ((d["HomeTeam"] == team) | (d["AwayTeam"] == team)).to_numpy()
        out[team] = (dates[mk], g1[mk], g2[mk])
    return out


def _share_app(logs, home, away, date, n=10):
    """Part de buts en 1re mi-temps pour ce match : même méthode que la carte
    (10 derniers matchs de chaque équipe, ramenés vers 45 % avec 20 buts fictifs)."""
    vals = []
    for team in (home, away):
        if team not in logs:
            return HALF_SHARE
        dates, g1, g2 = logs[team]
        k = int(np.searchsorted(dates, np.datetime64(date), side="left"))
        if k < 3:
            return HALF_SHARE
        a = max(0, k - n)
        s1, s2 = g1[a:k].sum(), g2[a:k].sum()
        vals.append((s1 + HALF_SHARE * HALF_PSEUDO) / (s1 + s2 + HALF_PSEUDO))
    return float(np.mean(vals))


# ----------------------------------------------------------------------
# 2. Prédictions « dans le passé »
# ----------------------------------------------------------------------
def oos_predictions(df, test_frac=0.4, block_days=60, min_train=600):
    df = df.sort_values("Date").reset_index(drop=True)
    has_half = "HTHG" in df.columns
    logs = _team_logs(df) if has_half else {}
    start_idx = max(int(len(df) * (1 - test_frac)), min_train)
    if start_idx >= len(df) - 100:
        raise ValueError("Pas assez de matchs pour corriger ce championnat.")
    rows = []
    bs, t_end = df["Date"].iloc[start_idx], df["Date"].max()
    while bs <= t_end:
        be = bs + pd.Timedelta(days=block_days)
        blk = df[(df["Date"] >= bs) & (df["Date"] < be)]
        if len(blk):
            model = DixonColes().fit(df[df["Date"] < bs], ref_date=bs - pd.Timedelta(days=1))
            for _, r in blk.iterrows():
                try:
                    m = model.score_matrix(r["HomeTeam"], r["AwayTeam"])
                    lh, la = model.rates(r["HomeTeam"], r["AwayTeam"])
                except Exception:
                    continue
                p1, pn, p2, p15, pgg = _probs(m)
                pick = int(np.argmax([p1, pn, p2]))
                ii, jj = np.indices(m.shape)
                mask = (ii > jj) if pick == 0 else ((ii == jj) if pick == 1 else (ii < jj))
                si, sj = np.unravel_index(np.argmax(np.where(mask, m, -1.0)), m.shape)
                mk = fc.market_from_row(r)
                ht_ok = has_half and pd.notna(r.get("HTHG")) and pd.notna(r.get("HTAG"))
                rows.append({
                    "date": r["Date"], "home": r["HomeTeam"], "away": r["AwayTeam"],
                    "hg": int(r["FTHG"]), "ag": int(r["FTAG"]), "lh": lh, "la": la,
                    "rho": float(model.rho),
                    "p1": p1, "pn": pn, "p2": p2, "p15": p15, "pgg": pgg,
                    "y": 0 if r["FTHG"] > r["FTAG"] else (1 if r["FTHG"] == r["FTAG"] else 2),
                    "o15": int(r["FTHG"] + r["FTAG"] >= 2),
                    "gg": int(r["FTHG"] > 0 and r["FTAG"] > 0),
                    "m1": np.nan if mk is None else mk[0],
                    "mn": np.nan if mk is None else mk[1],
                    "m2": np.nan if mk is None else mk[2],
                    "si": int(si), "sj": int(sj),
                    "h1": float(r["HTHG"]) if ht_ok else np.nan,
                    "a1": float(r["HTAG"]) if ht_ok else np.nan,
                    "s_app": _share_app(logs, r["HomeTeam"], r["AwayTeam"], r["Date"]),
                })
        bs = be
    out = pd.DataFrame(rows)
    if len(out) < 200:
        raise ValueError("Pas assez de prédictions pour corriger ce championnat.")
    return out


def _with_half(P):
    return P.dropna(subset=["h1", "a1"]) if "h1" in P.columns else P.iloc[0:0]


def league_share(P):
    """Part des buts marqués en 1re mi-temps dans ce championnat (tous les matchs connus)."""
    H = _with_half(P)
    tot = float((H["hg"] + H["ag"]).sum())
    if len(H) < 50 or tot <= 0:
        return HALF_SHARE
    return float((H["h1"] + H["a1"]).sum() / tot)
