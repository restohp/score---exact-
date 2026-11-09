"""Auto-correction des prédictions : comparer prédictions et résultats, puis corriger.

Principe (aucune triche) :
  1. on refait des prédictions « dans le passé » : chaque match est prédit avec les seuls
     matchs qui le précèdent (comme en vrai) ;
  2. on compare ces probabilités à ce qui s'est réellement passé ;
  3. on apprend une petite correction (trop de victoires à domicile ? pas assez de nuls ?
     trop de matchs à +1,5 but ?) et on l'applique aux prédictions futures.
La correction est volontairement prudente : elle est ramenée vers « ne rien changer »
tant qu'il y a peu de matchs, et elle est jugée sur des matchs qu'elle n'a pas vus.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import forme_core as fc
from dixon_coles import DixonColes

RIDGE = 10.0


# ----------------------------------------------------------------------
# 1. Prédictions « dans le passé »
# ----------------------------------------------------------------------
def oos_predictions(df, test_frac=0.4, block_days=60, min_train=600):
    df = df.sort_values("Date").reset_index(drop=True)
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
                p = DixonColes.probs_1x2(m)
                n = m.shape[0]
                tg = np.add.outer(np.arange(n), np.arange(n))
                mk = fc.market_from_row(r)
                rows.append({
                    "date": r["Date"], "home": r["HomeTeam"], "away": r["AwayTeam"],
                    "hg": int(r["FTHG"]), "ag": int(r["FTAG"]), "lh": lh, "la": la,
                    "p1": p[0], "pn": p[1], "p2": p[2],
                    "p15": float(m[tg >= 2].sum()),
                    "pgg": float(1 - m[0, :].sum() - m[:, 0].sum() + m[0, 0]),
                    "y": 0 if r["FTHG"] > r["FTAG"] else (1 if r["FTHG"] == r["FTAG"] else 2),
                    "o15": int(r["FTHG"] + r["FTAG"] >= 2),
                    "gg": int(r["FTHG"] > 0 and r["FTAG"] > 0),
                    "m1": np.nan if mk is None else mk[0],
                    "mn": np.nan if mk is None else mk[1],
                    "m2": np.nan if mk is None else mk[2],
                })
        bs = be
    out = pd.DataFrame(rows)
    if len(out) < 200:
        raise ValueError("Pas assez de prédictions pour corriger ce championnat.")
    return out


# ----------------------------------------------------------------------
# 2. Corrections
# ----------------------------------------------------------------------
def _lg(p):
    return np.log(np.clip(p, 1e-9, 1.0))


def apply_1x2(P, par):
    """P : tableau (n, 3). Température a + biais sur nul et extérieur."""
    a, bn, b2 = par
    z = a * _lg(np.asarray(P, float)) + np.array([0.0, bn, b2])
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def apply_bin(p, par):
    """Probabilité d'un événement oui/non : logit(p') = a * logit(p) + b."""
    a, b = par
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    z = a * np.log(p / (1 - p)) + b
    return 1.0 / (1.0 + np.exp(-z))


def _fit_1x2(P, y):
    n = len(y)

    def loss(par):
        Q = apply_1x2(P, par)
        ll = -np.mean(np.log(np.clip(Q[np.arange(n), y], 1e-12, 1)))
        return ll + (RIDGE / n) * ((par[0] - 1) ** 2 + par[1] ** 2 + par[2] ** 2)

    r = minimize(loss, x0=[1.0, 0.0, 0.0], method="L-BFGS-B",
                 bounds=[(0.6, 1.6), (-0.8, 0.8), (-0.8, 0.8)])
    return tuple(float(v) for v in r.x)


def _fit_bin(p, y):
    n = len(y)

    def loss(par):
        q = np.clip(apply_bin(p, par), 1e-12, 1 - 1e-12)
        ll = -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
        return ll + (RIDGE / n) * ((par[0] - 1) ** 2 + par[1] ** 2)

    r = minimize(loss, x0=[1.0, 0.0], method="L-BFGS-B", bounds=[(0.5, 1.8), (-1.0, 1.0)])
    return tuple(float(v) for v in r.x)


def fit_calibration(P):
    """P : DataFrame issu de oos_predictions. Renvoie les paramètres de correction."""
    X = P[["p1", "pn", "p2"]].to_numpy()
    return {
        "x12": _fit_1x2(X, P["y"].to_numpy()),
        "o15": _fit_bin(P["p15"].to_numpy(), P["o15"].to_numpy().astype(float)),
        "gg": _fit_bin(P["pgg"].to_numpy(), P["gg"].to_numpy().astype(float)),
        "n": int(len(P)),
    }


def correct(p1x2, p15, pgg, cal):
    """Applique la correction à un match. Renvoie (p1x2 corrigé, p15 corrigé, pgg corrigé)."""
    q = apply_1x2(np.asarray(p1x2, float)[None, :], cal["x12"])[0]
    return q, float(apply_bin(p15, cal["o15"])), float(apply_bin(pgg, cal["gg"]))


# ----------------------------------------------------------------------
# 3. Évaluation honnête : on corrige sur le début, on juge sur la fin
# ----------------------------------------------------------------------
def _ll1x2(Q, y):
    return -np.log(np.clip(Q[np.arange(len(y)), y], 1e-12, 1))


def _llbin(q, y):
    q = np.clip(q, 1e-12, 1 - 1e-12)
    return -(y * np.log(q) + (1 - y) * np.log(1 - q))


def _z(d):
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if len(d) > 2 and d.std() > 0 else float("nan")


def evaluate(P, fit_frac=0.6):
    P = P.sort_values("date").reset_index(drop=True)
    k = int(len(P) * fit_frac)
    A, B = P.iloc[:k], P.iloc[k:]
    cal = fit_calibration(A)
    X = B[["p1", "pn", "p2"]].to_numpy()
    yb = B["y"].to_numpy()
    Q = apply_1x2(X, cal["x12"])
    rows = []
    d = _ll1x2(Q, yb) - _ll1x2(X, yb)
    rows.append(("Résultat 1/N/2", _ll1x2(X, yb).mean(), _ll1x2(Q, yb).mean(), _z(d)))
    for name, col, ycol, key in (("+1,5 but", "p15", "o15", "o15"), ("Les deux marquent", "pgg", "gg", "gg")):
        p, y = B[col].to_numpy(), B[ycol].to_numpy().astype(float)
        q = apply_bin(p, cal[key])
        rows.append((name, _llbin(p, y).mean(), _llbin(q, y).mean(), _z(_llbin(q, y) - _llbin(p, y))))
    res = pd.DataFrame(rows, columns=["Marché", "Avant correction", "Après correction", "z (négatif = mieux)"])
    mk = B[["m1", "mn", "m2"]].to_numpy()
    ok = ~np.isnan(mk).any(axis=1)
    vs_market = None
    if ok.sum() > 50:
        vs_market = {"n": int(ok.sum()),
                     "avant": float(_ll1x2(X[ok], yb[ok]).mean()),
                     "apres": float(_ll1x2(Q[ok], yb[ok]).mean()),
                     "marche": float(_ll1x2(mk[ok], yb[ok]).mean())}
    return res, cal, vs_market, len(B)


def predicted_vs_observed(P):
    """Moyenne prévue contre fréquence réelle, pour chaque type de pari."""
    n = len(P)
    items = [
        ("Victoire domicile", P["p1"].mean(), (P["y"] == 0).mean()),
        ("Match nul", P["pn"].mean(), (P["y"] == 1).mean()),
        ("Victoire extérieur", P["p2"].mean(), (P["y"] == 2).mean()),
        ("+1,5 but", P["p15"].mean(), P["o15"].mean()),
        ("Les deux marquent", P["pgg"].mean(), P["gg"].mean()),
    ]
    out = pd.DataFrame(items, columns=["Pari", "Prévu", "Réel"])
    out["Écart (points)"] = (out["Prévu"] - out["Réel"]) * 100
    return out, n


def reliability(p, y, bins=5):
    """Découpe en groupes de même taille : probabilité prévue moyenne contre fréquence réelle."""
    d = pd.DataFrame({"p": np.asarray(p, float), "y": np.asarray(y, float)})
    d["g"] = pd.qcut(d["p"], bins, duplicates="drop")
    t = d.groupby("g", observed=True).agg(prevu=("p", "mean"), reel=("y", "mean"), n=("p", "size")).reset_index(drop=True)
    return t


# ----------------------------------------------------------------------
# 4. Revue match par match et équipes mal évaluées
# ----------------------------------------------------------------------
def hit_rates(P):
    """Taux de réussite des pronostics (le pronostic = le choix le plus probable)."""
    pick = P[["p1", "pn", "p2"]].to_numpy().argmax(axis=1)
    return {
        "Résultat 1/N/2": float((pick == P["y"].to_numpy()).mean()),
        "+1,5 but": float(((P["p15"] >= 0.5).astype(int) == P["o15"]).mean()),
        "Les deux marquent": float(((P["pgg"] >= 0.5).astype(int) == P["gg"]).mean()),
    }


def review_table(P, n=40):
    """Les n derniers matchs : vrai score, pronostic, et ✔ / ✘ pour chaque pari."""
    d = P.sort_values("date", ascending=False).head(n)
    out = []
    for _, r in d.iterrows():
        pick = int(np.argmax([r["p1"], r["pn"], r["p2"]]))
        over = r["p15"] >= 0.5
        gg = r["pgg"] >= 0.5
        out.append({
            "Date": r["date"].strftime("%d/%m/%Y"),
            "Match": f"{r['home']} - {r['away']}",
            "Score réel": f"{r['hg']}-{r['ag']}",
            "1/N/2": ["1", "N", "2"][pick] + (" ✔" if pick == r["y"] else " ✘"),
            "+1,5 but": ("Oui" if over else "Non") + (" ✔" if int(over) == r["o15"] else " ✘"),
            "GG": ("Oui" if gg else "Non") + (" ✔" if int(gg) == r["gg"] else " ✘"),
        })
    return pd.DataFrame(out)


def team_errors(P, min_n=8):
    """Par équipe : buts marqués / encaissés réels moins buts attendus par l'appli (par match)."""
    h = pd.DataFrame({"team": P["home"], "gf": P["hg"], "ef": P["lh"], "ga": P["ag"], "ea": P["la"]})
    a = pd.DataFrame({"team": P["away"], "gf": P["ag"], "ef": P["la"], "ga": P["hg"], "ea": P["lh"]})
    L = pd.concat([h, a], ignore_index=True)
    rows = []
    for team, g in L.groupby("team"):
        if len(g) < min_n:
            continue
        d_for = (g["gf"] - g["ef"]).to_numpy()
        d_ag = (g["ga"] - g["ea"]).to_numpy()
        rows.append({
            "Équipe": team, "Matchs": len(g),
            "Buts marqués (réel - prévu)": float(d_for.mean()),
            "z marqués": _z(d_for),
            "Buts encaissés (réel - prévu)": float(d_ag.mean()),
            "z encaissés": _z(d_ag),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["écart max"] = np.maximum(out["z marqués"].abs().fillna(0), out["z encaissés"].abs().fillna(0))
    return out.sort_values("écart max", ascending=False).drop(columns="écart max").reset_index(drop=True)
