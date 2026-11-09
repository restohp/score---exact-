"""Auto-correction des prédictions : comparer prédictions et résultats, puis corriger.

Principe (aucune triche) :
  1. on refait des prédictions « dans le passé » : chaque match est prédit avec les seuls
     matchs qui le précèdent (comme en vrai) ;
  2. on compare ces probabilités à ce qui s'est réellement passé (score final ET mi-temps) ;
  3. on apprend des corrections prudentes :
       - globale : trop de victoires à domicile ? pas assez de nuls ? trop de +1,5 but ?
       - par équipe : l'appli sous-estime / surestime l'attaque ou la défense de certaines équipes ;
       - mi-temps : part des buts marqués en 1re mi-temps dans ce championnat ;
  4. chaque correction est jugée sur des matchs qu'elle n'a pas vus, et n'est activée
     que si elle améliore nettement le résultat.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

import dixon_coles as dc
import forme_core as fc
from dixon_coles import DixonColes

RIDGE = 10.0
TEAM_K = 30.0        # « buts fictifs » qui ramènent la correction d'une équipe vers 0
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
    cal["team"] = team
    cal["z_team"] = z_all
    cal["share"] = league_share(P)
    return cal


# ----------------------------------------------------------------------
# 6. Revue match par match et équipes mal évaluées
# ----------------------------------------------------------------------
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
    return d.groupby("g", observed=True).agg(prevu=("p", "mean"), reel=("y", "mean"),
                                             n=("p", "size")).reset_index(drop=True)


def hit_rates(P):
    """Taux de réussite des pronostics (le pronostic = le choix le plus probable)."""
    pick = P[["p1", "pn", "p2"]].to_numpy().argmax(axis=1)
    return {
        "Résultat 1/N/2": float((pick == P["y"].to_numpy()).mean()),
        "+1,5 but": float(((P["p15"] >= 0.5).astype(int) == P["o15"]).mean()),
        "Les deux marquent": float(((P["pgg"] >= 0.5).astype(int) == P["gg"]).mean()),
    }


def _half_pred(P):
    """Scores prédits (comme sur la carte) pour la 1re et la 2e mi-temps."""
    h1p, a1p, h2p, a2p = [], [], [], []
    for si, sj, s in zip(P["si"], P["sj"], P["s_app"]):
        (a, b), (c, d) = half_scores(int(si), int(sj), float(s))
        h1p.append(a); a1p.append(b); h2p.append(c); a2p.append(d)
    return (np.array(h1p), np.array(a1p), np.array(h2p), np.array(a2p))


def _sign(a, b):
    return np.where(a > b, 0, np.where(a == b, 1, 2))


def review_table(P, n=40):
    """Les n derniers matchs : vrai score, pronostic, et ✔ / ✘ pour chaque pari (mi-temps comprises)."""
    d = P.sort_values("date", ascending=False).head(n)
    h1p, a1p, h2p, a2p = _half_pred(d)
    out = []
    for k, (_, r) in enumerate(d.iterrows()):
        pick = int(np.argmax([r["p1"], r["pn"], r["p2"]]))
        over = r["p15"] >= 0.5
        gg = r["pgg"] >= 0.5
        row = {
            "Date": r["date"].strftime("%d/%m/%Y"),
            "Match": f"{r['home']} - {r['away']}",
            "Score réel": f"{r['hg']}-{r['ag']}",
            "1/N/2": ["1", "N", "2"][pick] + (" ✔" if pick == r["y"] else " ✘"),
            "+1,5 but": ("Oui" if over else "Non") + (" ✔" if int(over) == r["o15"] else " ✘"),
            "GG": ("Oui" if gg else "Non") + (" ✔" if int(gg) == r["gg"] else " ✘"),
        }
        if pd.notna(r["h1"]):
            r1, b1 = int(r["h1"]), int(r["a1"])
            r2, b2 = int(r["hg"]) - r1, int(r["ag"]) - b1
            row["1re MT (prévu → réel)"] = (f"{h1p[k]}-{a1p[k]} → {r1}-{b1}"
                                            + (" ✔" if (h1p[k], a1p[k]) == (r1, b1) else " ✘"))
            row["2e MT (prévu → réel)"] = (f"{h2p[k]}-{a2p[k]} → {r2}-{b2}"
                                           + (" ✔" if (h2p[k], a2p[k]) == (r2, b2) else " ✘"))
        out.append(row)
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


def team_factors_table(team, top=8):
    """Les équipes que la correction par équipe change le plus (en % de buts)."""
    rows = []
    for t in team["for"]:
        rows.append({"Équipe": t,
                     "Attaque (buts)": (team["for"][t] - 1) * 100,
                     "Défense (buts encaissés)": (team["ag"].get(t, 1.0) - 1) * 100})
    d = pd.DataFrame(rows)
    d["_m"] = np.maximum(d["Attaque (buts)"].abs(), d["Défense (buts encaissés)"].abs())
    return d.sort_values("_m", ascending=False).drop(columns="_m").head(top).reset_index(drop=True)


# ----------------------------------------------------------------------
# 7. Mi-temps
# ----------------------------------------------------------------------
def _with_half(P):
    return P.dropna(subset=["h1", "a1"]) if "h1" in P.columns else P.iloc[0:0]


def league_share(P):
    """Part des buts marqués en 1re mi-temps dans ce championnat (tous les matchs connus)."""
    H = _with_half(P)
    tot = float((H["hg"] + H["ag"]).sum())
    if len(H) < 50 or tot <= 0:
        return HALF_SHARE
    return float((H["h1"] + H["a1"]).sum() / tot)


def half_hit_rates(P):
    """Réussite des scores de mi-temps prédits (comme sur la carte), comparée au pari
    « toujours le même choix » (le score ou résultat le plus fréquent)."""
    H = _with_half(P)
    if len(H) < 30:
        return None
    h1p, a1p, h2p, a2p = _half_pred(H)
    r1h, r1a = H["h1"].to_numpy(int), H["a1"].to_numpy(int)
    r2h, r2a = H["hg"].to_numpy(int) - r1h, H["ag"].to_numpy(int) - r1a

    def exact(ph, pa, rh, ra):
        return float(((ph == rh) & (pa == ra)).mean())

    def common_score(rh, ra):
        return float(pd.Series(list(zip(rh, ra))).value_counts(normalize=True).iloc[0])

    def common_res(rh, ra):
        return float(pd.Series(_sign(rh, ra)).value_counts(normalize=True).iloc[0])

    rows = [
        ("1re MT : score exact", exact(h1p, a1p, r1h, r1a), common_score(r1h, r1a)),
        ("1re MT : résultat 1/N/2", float((_sign(h1p, a1p) == _sign(r1h, r1a)).mean()), common_res(r1h, r1a)),
        ("2e MT : score exact", exact(h2p, a2p, r2h, r2a), common_score(r2h, r2a)),
        ("2e MT : résultat 1/N/2", float((_sign(h2p, a2p) == _sign(r2h, r2a)).mean()), common_res(r2h, r2a)),
    ]
    return pd.DataFrame(rows, columns=["Pari", "Réussite de l'appli", "Toujours le même choix"]), len(H)


def _ll_goals(lam, k):
    return -poisson.logpmf(k, np.clip(lam, 1e-3, None))


def evaluate_half(P, fit_frac=0.6):
    """Quelle part de buts en 1re mi-temps prédit le mieux les buts de chaque mi-temps ?

    Trois méthodes, jugées sur les matchs les plus récents (perte = log-vraisemblance négative
    des buts réels, plus petit = meilleur) :
      - part fixe 45 % ;
      - part du championnat, apprise sur les 60 % de matchs les plus anciens ;
      - part par équipe (10 derniers matchs), comme sur la carte.
    """
    H = _with_half(P).sort_values("date").reset_index(drop=True)
    if len(H) < 100:
        return None
    k = int(len(H) * fit_frac)
    A, B = H.iloc[:k], H.iloc[k:]
    s_lg = league_share(A)
    r1h, r1a = B["h1"].to_numpy(), B["a1"].to_numpy()
    r2h, r2a = B["hg"].to_numpy() - r1h, B["ag"].to_numpy() - r1a
    lh, la = B["lh"].to_numpy(), B["la"].to_numpy()

    def loss(s, half):
        s = np.broadcast_to(np.asarray(s, float), lh.shape)
        if half == 1:
            return _ll_goals(lh * s, r1h) + _ll_goals(la * s, r1a)
        return _ll_goals(lh * (1 - s), r2h) + _ll_goals(la * (1 - s), r2a)

    methods = [("Part fixe 45 %", HALF_SHARE),
               (f"Part du championnat ({s_lg:.1%})", s_lg),
               ("Part par équipe (carte)", B["s_app"].to_numpy())]
    rows = []
    for half, name in ((1, "Buts de la 1re mi-temps"), (2, "Buts de la 2e mi-temps")):
        base = loss(methods[0][1], half)
        row = {"Mi-temps": name}
        for label, s in methods:
            row[label] = float(loss(s, half).mean())
        row["z championnat"] = _z(loss(methods[1][1], half) - base)
        row["z par équipe"] = _z(loss(methods[2][1], half) - base)
        rows.append(row)
    return pd.DataFrame(rows), s_lg, len(B)
