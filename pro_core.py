"""Modèle pro : tout ce qui a été validé, réuni dans un seul modèle (aucun fichier existant modifié).

- force des équipes apprise sur un mélange 50 % buts réels / 50 % buts estimés par les tirs
  (utilisé seulement si le test ne le trouve pas nettement moins bon que les buts seuls) ;
- auto-correction (1/N/2, +1,5 but, GG) apprise sur ce modèle, gardée seulement là où elle a
  nettement amélioré le test ;
- correction par équipe : non (jamais validée).
"""
import numpy as np
import pandas as pd

import ac_data as ad
import ac_fit as af
import forme_core as fc
import shots_model as sm
from dixon_coles import DixonColes
from shots_model import DixonColesShots

BLEND = 0.5      # poids des vrais buts dans le mélange


def oos_pro(df, w, test_frac=0.4, block_days=60, min_train=600):
    """Prédictions dans le passé (même format que ac_data.oos_predictions) avec le modèle de poids w."""
    df = df.sort_values("Date").reset_index(drop=True)
    has_half = "HTHG" in df.columns
    logs = ad._team_logs(df) if has_half else {}
    start_idx = max(int(len(df) * (1 - test_frac)), min_train)
    if start_idx >= len(df) - 100:
        raise ValueError("Pas assez de matchs pour ce championnat.")
    rows = []
    bs, t_end = df["Date"].iloc[start_idx], df["Date"].max()
    while bs <= t_end:
        be = bs + pd.Timedelta(days=block_days)
        blk = df[(df["Date"] >= bs) & (df["Date"] < be)]
        if len(blk):
            model = DixonColesShots(w=w).fit(df[df["Date"] < bs], ref_date=bs - pd.Timedelta(days=1))
            for _, r in blk.iterrows():
                try:
                    m = model.score_matrix(r["HomeTeam"], r["AwayTeam"])
                    lh, la = model.rates(r["HomeTeam"], r["AwayTeam"])
                except Exception:
                    continue
                p1, pn, p2, p15, pgg = ad._probs(m)
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
                    "s_app": ad._share_app(logs, r["HomeTeam"], r["AwayTeam"], r["Date"]),
                })
        bs = be
    out = pd.DataFrame(rows)
    if len(out) < 200:
        raise ValueError("Pas assez de prédictions pour ce championnat.")
    return out


def _ll(P):
    Q = P[["p1", "pn", "p2"]].to_numpy()
    return -np.log(np.clip(Q[np.arange(len(P)), P["y"].to_numpy()], 1e-12, 1))


def compare_versions(Pg, Ps):
    """Mêmes matchs, buts seuls contre mélange : (log-loss buts, log-loss mélange, z, n)."""
    key = ["date", "home", "away"]
    a = Pg[key + ["p1", "pn", "p2", "y"]].merge(Ps[key + ["p1", "pn", "p2"]], on=key,
                                                suffixes=("_g", "_s"))
    Qg = a[["p1_g", "pn_g", "p2_g"]].to_numpy()
    Qs = a[["p1_s", "pn_s", "p2_s"]].to_numpy()
    y = a["y"].to_numpy()
    lg = -np.log(np.clip(Qg[np.arange(len(a)), y], 1e-12, 1))
    ls = -np.log(np.clip(Qs[np.arange(len(a)), y], 1e-12, 1))
    return float(lg.mean()), float(ls.mean()), af._z(ls - lg), len(a)


def build(code, n_seasons=5):
    """Tout ce que le modèle pro apprend d'un championnat (long : 2 à 5 minutes)."""
    df = ad.load_data(code, n_seasons=n_seasons)
    cov = sm.shots_coverage(df)
    Pg = oos_pro(df, 1.0)
    info = {"cov": cov, "z_shots": float("nan"), "ll_goals": float(_ll(Pg).mean()),
            "ll_blend": float("nan"), "w": 1.0, "n": len(Pg)}
    P = Pg
    if cov >= 0.7:
        Ps = oos_pro(df, BLEND)
        lg, ls, z, n = compare_versions(Pg, Ps)
        info.update({"z_shots": z, "ll_goals": lg, "ll_blend": ls, "n": n})
        if np.isfinite(z) and z <= 2.0:          # jamais nettement moins bon que les buts seuls
            info["w"] = BLEND
            P = Ps
    cal = af.fit_full(P)
    mk = P[["m1", "mn", "m2"]].to_numpy()
    ok = ~np.isnan(mk).any(axis=1)
    if ok.sum() > 100:
        y = P["y"].to_numpy()
        info["ll_market"] = float((-np.log(np.clip(mk[ok][np.arange(ok.sum()), y[ok]], 1e-12, 1))).mean())
        info["ll_pro_market_rows"] = float(_ll(P)[ok].mean())
    return {"df": df, "cal": cal, "info": info}


def fit_model(df, w, cutoff):
    cutoff = pd.Timestamp(cutoff)
    train = df[df["Date"] < cutoff]
    if len(train) < 200:
        raise ValueError("Pas assez de matchs avant cette date pour entraîner le modèle.")
    return DixonColesShots(w=w).fit(train, ref_date=cutoff - pd.Timedelta(days=1))


def predict(model, cal, home, away):
    """Pronostic du modèle pro pour un match : probabilités corrigées, pronostic, score prévu."""
    m = af.adjusted_matrix(model, home, away, cal)
    n = m.shape[0]
    tg = np.add.outer(np.arange(n), np.arange(n))
    p1x2 = DixonColes.probs_1x2(m)
    p15 = float(m[tg >= 2].sum())
    pgg = float(1 - m[0, :].sum() - m[:, 0].sum() + m[0, 0])
    p1x2, p15, pgg = af.correct(p1x2, p15, pgg, cal)
    pick = int(np.argmax(p1x2))
    ii, jj = np.indices(m.shape)
    mask = (ii > jj) if pick == 0 else ((ii == jj) if pick == 1 else (ii < jj))
    i, j = np.unravel_index(np.argmax(np.where(mask, m, -1.0)), m.shape)
    return {"p": np.asarray(p1x2, float), "pick": pick, "score": (int(i), int(j)),
            "p15": float(p15), "pgg": float(pgg)}
