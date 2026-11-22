"""Test : est-ce que les tirs aident à mieux estimer la force des équipes ?

Même modèle que Dixon-Coles, mais la force des équipes est apprise sur un mélange
« buts réels + buts estimés à partir des tirs » (w = poids des vrais buts).
w = 1 reproduit exactement le modèle actuel. Le test est honnête : chaque match est prédit avec
les seuls matchs qui le précèdent, puis on compare le log-loss (plus petit = meilleur).
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls

import dixon_coles as dc
import forme_core as fc
from dixon_coles import DixonColes

SHOT_COLS = ["HS", "AS", "HST", "AST"]


def shots_coverage(df):
    if not set(SHOT_COLS).issubset(df.columns):
        return 0.0
    return float(df[SHOT_COLS].notna().all(axis=1).mean())


def xg_coefficients(data):
    """Buts ≈ a * tirs cadrés + b * tirs non cadrés (régression sans constante, a,b >= 0)."""
    d = data.dropna(subset=SHOT_COLS)
    if len(d) < 100:
        return 0.30, 0.03
    rows, y = [], []
    for sh, st_, g in (("HS", "HST", "FTHG"), ("AS", "AST", "FTAG")):
        on = d[st_].to_numpy(float)
        off = np.clip(d[sh].to_numpy(float) - on, 0, None)
        rows.append(np.column_stack([on, off]))
        y.append(d[g].to_numpy(float))
    coef, _ = nnls(np.vstack(rows), np.concatenate(y))
    scale = np.concatenate(y).mean() / max((np.vstack(rows) @ coef).mean(), 1e-9)
    return float(coef[0] * scale), float(coef[1] * scale)


def blended_targets(data, w):
    """Buts à expliquer par le modèle : w * vrais buts + (1 - w) * buts estimés par les tirs."""
    a, b = xg_coefficients(data)
    out = []
    for sh, st_, g in (("HS", "HST", "FTHG"), ("AS", "AST", "FTAG")):
        goals = data[g].to_numpy(float)
        on = data[st_].to_numpy(float)
        off = np.clip(data[sh].to_numpy(float) - on, 0, None)
        xg = a * on + b * off
        xg = np.where(np.isnan(xg), goals, xg)          # pas de tirs : on garde les vrais buts
        out.append(w * goals + (1 - w) * xg)
    return out[0], out[1]


class DixonColesShots(DixonColes):
    def __init__(self, xi=0.0018, w=0.5):
        super().__init__(xi)
        self.w = float(w)

    def fit(self, df, ref_date=None):
        data = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
        if len(data) < 50:
            raise ValueError("Moins de 50 matchs : historique insuffisant.")
        data["FTHG"] = data["FTHG"].astype(int)
        data["FTAG"] = data["FTAG"].astype(int)
        ref_date = pd.Timestamp(ref_date if ref_date is not None else data["Date"].max())

        teams = sorted(set(data["HomeTeam"]) | set(data["AwayTeam"]))
        index = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        home_idx = data["HomeTeam"].map(index).to_numpy()
        away_idx = data["AwayTeam"].map(index).to_numpy()
        hg, ag = data["FTHG"].to_numpy(), data["FTAG"].to_numpy()          # vrais buts (corrélation)
        if self.w >= 1.0:
            yh, ya = hg.astype(float), ag.astype(float)
        else:
            yh, ya = blended_targets(data, self.w)
        days = (ref_date - data["Date"]).dt.days.clip(lower=0).to_numpy()
        weights = np.exp(-self.xi * days)
        m00 = (hg == 0) & (ag == 0)
        m01 = (hg == 0) & (ag == 1)
        m10 = (hg == 1) & (ag == 0)
        m11 = (hg == 1) & (ag == 1)

        def unpack(p):
            return (p[:n] - p[:n].mean(), p[n:2 * n] - p[n:2 * n].mean(), p[2 * n], p[2 * n + 1], p[2 * n + 2])

        def nll(p):
            att, dfn, home, rho, mu = unpack(p)
            lh = np.exp(np.clip(mu + att[home_idx] + dfn[away_idx] + home, -5, 4))
            la = np.exp(np.clip(mu + att[away_idx] + dfn[home_idx], -5, 4))
            tau = np.ones(len(data))
            tau[m00] = 1 - lh[m00] * la[m00] * rho
            tau[m01] = 1 + lh[m01] * rho
            tau[m10] = 1 + la[m10] * rho
            tau[m11] = 1 - rho
            tau = np.clip(tau, 1e-8, None)
            ll = np.log(tau) + yh * np.log(lh) - lh + ya * np.log(la) - la
            return -float(np.sum(weights * ll)) + 0.001 * np.square(p[:2 * n]).sum()

        x0 = np.zeros(2 * n + 3)
        x0[2 * n], x0[2 * n + 1], x0[2 * n + 2] = 0.2, -0.05, 0.25
        bounds = [(None, None)] * (2 * n + 1) + [(-0.25, 0.25)] + [(None, None)]
        res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 1000})
        if not np.isfinite(res.fun):
            raise RuntimeError(f"Échec du modèle : {res.message}")
        self.teams, self.idx = teams, index
        self.att, self.defense, self.home, self.rho, self.mu = unpack(res.x)
        self.optimization_message = str(res.message)
        return self


# ----------------------------------------------------------------------
def _probs(m):
    n = m.shape[0]
    tg = np.add.outer(np.arange(n), np.arange(n))
    p = DixonColes.probs_1x2(m)
    return (p[0], p[1], p[2], float(m[tg >= 2].sum()),
            float(1 - m[0, :].sum() - m[:, 0].sum() + m[0, 0]))


def _tag(w):
    return f"w{int(round(w * 100))}"


def run_experiment(df, ws=(1.0, 0.5, 0.0), test_frac=0.4, block_days=90, min_train=600,
                   progress=None):
    """Prédictions dans le passé pour chaque version du modèle. Renvoie un DataFrame."""
    df = df.sort_values("Date").reset_index(drop=True)
    if shots_coverage(df) < 0.7:
        raise ValueError("Ce championnat n'a pas assez de données de tirs dans les fichiers gratuits.")
    start_idx = max(int(len(df) * (1 - test_frac)), min_train)
    if start_idx >= len(df) - 100:
        raise ValueError("Pas assez de matchs pour ce test.")
    starts, bs, t_end = [], df["Date"].iloc[start_idx], df["Date"].max()
    while bs <= t_end:
        starts.append(bs)
        bs = bs + pd.Timedelta(days=block_days)
    rows = []
    for k, bs in enumerate(starts):
        be = bs + pd.Timedelta(days=block_days)
        blk = df[(df["Date"] >= bs) & (df["Date"] < be)]
        if len(blk):
            train = df[df["Date"] < bs]
            ref = bs - pd.Timedelta(days=1)
            models = {w: DixonColesShots(w=w).fit(train, ref_date=ref) for w in ws}
            for _, r in blk.iterrows():
                try:
                    probs = {w: _probs(models[w].score_matrix(r["HomeTeam"], r["AwayTeam"])) for w in ws}
                except Exception:
                    continue
                mk = fc.market_from_row(r)
                row = {"date": r["Date"],
                       "y": 0 if r["FTHG"] > r["FTAG"] else (1 if r["FTHG"] == r["FTAG"] else 2),
                       "o15": int(r["FTHG"] + r["FTAG"] >= 2),
                       "gg": int(r["FTHG"] > 0 and r["FTAG"] > 0),
                       "m1": np.nan if mk is None else mk[0],
                       "mn": np.nan if mk is None else mk[1],
                       "m2": np.nan if mk is None else mk[2]}
                for w in ws:
                    for name, v in zip(("p1", "pn", "p2", "p15", "pgg"), probs[w]):
                        row[f"{_tag(w)}_{name}"] = v
                rows.append(row)
        if progress:
            progress((k + 1) / len(starts))
    out = pd.DataFrame(rows)
    if len(out) < 200:
        raise ValueError("Pas assez de prédictions pour ce test.")
    return out


def _ll3(Q, y):
    return -np.log(np.clip(Q[np.arange(len(y)), y], 1e-12, 1))


def _llb(q, y):
    q = np.clip(q, 1e-12, 1 - 1e-12)
    return -(y * np.log(q) + (1 - y) * np.log(1 - q))


def _z(d):
    d = np.asarray(d, float)
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if len(d) > 2 and d.std() > 0 else float("nan")


def evaluate_experiment(P, ws=(1.0, 0.5, 0.0), base=1.0):
    """Tableau : log-loss de chaque version + z contre la version actuelle (négatif = mieux)."""
    y, o15, gg = P["y"].to_numpy(), P["o15"].to_numpy(float), P["gg"].to_numpy(float)
    loss = {}
    for w in ws:
        t = _tag(w)
        Q = P[[f"{t}_p1", f"{t}_pn", f"{t}_p2"]].to_numpy()
        loss[w] = (_ll3(Q, y), _llb(P[f"{t}_p15"].to_numpy(), o15), _llb(P[f"{t}_pgg"].to_numpy(), gg))
    names = {1.0: "Buts seuls (modèle actuel)", 0.0: "Tirs seuls"}
    rows = []
    for w in ws:
        label = names.get(w, f"Mélange ({int(round(w * 100))} % buts, {int(round((1 - w) * 100))} % tirs)")
        l1, l2, l3 = loss[w]
        b1, b2, b3 = loss[base]
        rows.append((label, l1.mean(), l2.mean(), l3.mean(),
                     np.nan if w == base else _z(l1 - b1),
                     np.nan if w == base else _z(l2 - b2),
                     np.nan if w == base else _z(l3 - b3)))
    tab = pd.DataFrame(rows, columns=["Version", "1/N/2", "+1,5 but", "GG",
                                      "z 1/N/2", "z +1,5", "z GG"])
    mk = P[["m1", "mn", "m2"]].to_numpy()
    ok = ~np.isnan(mk).any(axis=1)
    market = None
    if ok.sum() > 100:
        lm = _ll3(mk[ok], y[ok])
        market = {"n": int(ok.sum()), "marche": float(lm.mean()),
                  "versions": {w: float(loss[w][0][ok].mean()) for w in ws},
                  "z": {w: _z(loss[w][0][ok] - lm) for w in ws}}
    return tab, market, len(P)
