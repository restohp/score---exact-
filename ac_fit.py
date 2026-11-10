"""Auto-correction (2/3) : corrections (globale, par équipe) et évaluation honnête."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ac_data import HALF_SHARE, _probs, league_share, matrix_from_rates

RIDGE = 10.0
TEAM_K = 30.0        # « buts fictifs » qui ramènent la correction d'une équipe vers 0


# ----------------------------------------------------------------------
# 3. Corrections globales (température / décalage)
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
    """Correction globale seule. P : DataFrame issu de oos_predictions (ou adjust_rates)."""
    X = P[["p1", "pn", "p2"]].to_numpy()
    return {
        "x12": _fit_1x2(X, P["y"].to_numpy()),
        "o15": _fit_bin(P["p15"].to_numpy(), P["o15"].to_numpy().astype(float)),
        "gg": _fit_bin(P["pgg"].to_numpy(), P["gg"].to_numpy().astype(float)),
        "n": int(len(P)),
    }


def correct(p1x2, p15, pgg, cal):
    """Applique la correction globale à un match. Renvoie (p1x2, p15, pgg) corrigés."""
    q = apply_1x2(np.asarray(p1x2, float)[None, :], cal["x12"])[0]
    return q, float(apply_bin(p15, cal["o15"])), float(apply_bin(pgg, cal["gg"]))


# ----------------------------------------------------------------------
# 4. Correction par équipe (attaque et défense)
# ----------------------------------------------------------------------
def fit_team(P, k=TEAM_K):
    """Facteur par équipe = (buts réels + k) / (buts attendus + k), séparément en attaque et en
    défense. Les k « buts fictifs » ramènent vers 1 (aucune correction) les équipes peu observées."""
    h = pd.DataFrame({"t": P["home"], "gf": P["hg"], "ef": P["lh"], "ga": P["ag"], "ea": P["la"]})
    a = pd.DataFrame({"t": P["away"], "gf": P["ag"], "ef": P["la"], "ga": P["hg"], "ea": P["lh"]})
    g = pd.concat([h, a], ignore_index=True).groupby("t").sum()
    f_for = ((g["gf"] + k) / (g["ef"] + k)).to_dict()
    f_ag = ((g["ga"] + k) / (g["ea"] + k)).to_dict()
    return {"for": {t: float(v) for t, v in f_for.items()},
            "ag": {t: float(v) for t, v in f_ag.items()}}


def adjusted_rates(lh, la, home, away, team):
    ff, fa = team["for"], team["ag"]
    return (lh * ff.get(home, 1.0) * fa.get(away, 1.0),
            la * ff.get(away, 1.0) * fa.get(home, 1.0))


def adjust_rates(P, team):
    """Recalcule toutes les probabilités avec les taux de buts corrigés par équipe."""
    out = P.copy()
    cols = {"p1": [], "pn": [], "p2": [], "p15": [], "pgg": [], "lh": [], "la": []}
    for _, r in P.iterrows():
        lh, la = adjusted_rates(r["lh"], r["la"], r["home"], r["away"], team)
        p = _probs(matrix_from_rates(lh, la, r["rho"]))
        for key, v in zip(("p1", "pn", "p2", "p15", "pgg"), p):
            cols[key].append(v)
        cols["lh"].append(lh)
        cols["la"].append(la)
    for key, v in cols.items():
        out[key] = v
    return out


def adjusted_matrix(model, home, away, cal=None):
    """Matrice de scores d'un match, avec la correction par équipe si elle est activée."""
    if cal is not None and cal.get("team"):
        lh, la = model.rates(home, away)
        lh, la = adjusted_rates(lh, la, home, away, cal["team"])
        return matrix_from_rates(lh, la, model.rho)
    return model.score_matrix(home, away)


# ----------------------------------------------------------------------
# 5. Évaluation honnête : on apprend sur le début, on juge sur la fin
# ----------------------------------------------------------------------
def _ll1x2(Q, y):
    return -np.log(np.clip(Q[np.arange(len(y)), y], 1e-12, 1))


def _llbin(q, y):
    q = np.clip(q, 1e-12, 1 - 1e-12)
    return -(y * np.log(q) + (1 - y) * np.log(1 - q))


def _z(d):
    d = np.asarray(d, float)
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if len(d) > 2 and d.std() > 0 else float("nan")


def _losses(Pd, cal=None):
    """Pertes par match (1/N/2, +1,5, GG), avec ou sans correction globale."""
    X = Pd[["p1", "pn", "p2"]].to_numpy()
    y = Pd["y"].to_numpy()
    o15 = Pd["o15"].to_numpy().astype(float)
    gg = Pd["gg"].to_numpy().astype(float)
    if cal is None:
        return _ll1x2(X, y), _llbin(Pd["p15"].to_numpy(), o15), _llbin(Pd["pgg"].to_numpy(), gg)
    Q = apply_1x2(X, cal["x12"])
    return (_ll1x2(Q, y), _llbin(apply_bin(Pd["p15"].to_numpy(), cal["o15"]), o15),
            _llbin(apply_bin(Pd["pgg"].to_numpy(), cal["gg"]), gg))


def evaluate(P, fit_frac=0.6):
    P = P.sort_values("date").reset_index(drop=True)
    k = int(len(P) * fit_frac)
    A, B = P.iloc[:k], P.iloc[k:]
    cal = fit_calibration(A)
    raw = _losses(B)
    fix = _losses(B, cal)
    names = ("Résultat 1/N/2", "+1,5 but", "Les deux marquent")
    rows = [(nm, r.mean(), f.mean(), _z(f - r)) for nm, r, f in zip(names, raw, fix)]
    res = pd.DataFrame(rows, columns=["Marché", "Avant correction", "Après correction", "z (négatif = mieux)"])
    X = B[["p1", "pn", "p2"]].to_numpy()
    yb = B["y"].to_numpy()
    Q = apply_1x2(X, cal["x12"])
    mk = B[["m1", "mn", "m2"]].to_numpy()
    ok = ~np.isnan(mk).any(axis=1)
    vs_market = None
    if ok.sum() > 50:
        vs_market = {"n": int(ok.sum()),
                     "avant": float(_ll1x2(X[ok], yb[ok]).mean()),
                     "apres": float(_ll1x2(Q[ok], yb[ok]).mean()),
                     "marche": float(_ll1x2(mk[ok], yb[ok]).mean())}
    return res, cal, vs_market, len(B)


def evaluate_team(P, fit_frac=0.6):
    """Compare, sur les matchs les plus récents : brut, correction globale seule,
    correction par équipe seule, correction par équipe + globale.

    Renvoie (tableau, z global, n). Le z global juge « par équipe + globale » contre
    « globale seule » sur la somme des trois paris (négatif = la correction par équipe aide).
    """
    P = P.sort_values("date").reset_index(drop=True)
    k = int(len(P) * fit_frac)
    A, B = P.iloc[:k], P.iloc[k:]
    team = fit_team(A)
    A2, B2 = adjust_rates(A, team), adjust_rates(B, team)
    cal_g = fit_calibration(A)
    cal_tg = fit_calibration(A2)
    raw = _losses(B)
    glob = _losses(B, cal_g)
    team_only = _losses(B2)
    team_glob = _losses(B2, cal_tg)
    names = ("Résultat 1/N/2", "+1,5 but", "Les deux marquent")
    rows = []
    for nm, r, g, t, tg in zip(names, raw, glob, team_only, team_glob):
        rows.append((nm, r.mean(), g.mean(), t.mean(), tg.mean(), _z(tg - g)))
    tab = pd.DataFrame(rows, columns=["Marché", "Brut", "Globale seule", "Équipes seules",
                                      "Équipes + globale", "z (équipes + globale contre globale)"])
    d_all = sum(tg - g for g, tg in zip(glob, team_glob))
    return tab, _z(d_all), len(B)


def fit_full(P):
    """Correction finale utilisée par la page « Cartes ».

    - la correction par équipe n'est activée que si elle améliore nettement (z < -2) sur des
      matchs qu'elle n'a pas vus ;
    - la correction globale est apprise ensuite sur les probabilités déjà corrigées par équipe ;
    - chaque correction globale (1/N/2, +1,5 but, GG) n'est gardée que si elle a nettement
      amélioré le test ; sinon elle est remplacée par « aucune correction » ;
    - la part de buts en 1re mi-temps est celle observée dans le championnat.
    """
    try:
        _, z_all, _ = evaluate_team(P)
    except Exception:
        z_all = float("nan")
    use_team = bool(np.isfinite(z_all) and z_all < -2.0)
    team = fit_team(P) if use_team else None
    P2 = adjust_rates(P, team) if use_team else P
    cal = fit_calibration(P2)
    neutral = {"x12": (1.0, 0.0, 0.0), "o15": (1.0, 0.0), "gg": (1.0, 0.0)}
    keys = {"Résultat 1/N/2": "x12", "+1,5 but": "o15", "Les deux marquent": "gg"}
    used = {"x12": False, "o15": False, "gg": False}
    try:
        res = evaluate(P2)[0]
        for _, r in res.iterrows():
            k = keys[r["Marché"]]
            z = r["z (négatif = mieux)"]
            if np.isfinite(z) and z < -2.0:
                used[k] = True
    except Exception:
        pass
    for k, ok in used.items():
        if not ok:
            cal[k] = neutral[k]
    cal["used"] = used
    cal["team"] = team
    cal["z_team"] = z_all
    cal["share"] = league_share(P)
    return cal
