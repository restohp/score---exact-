"""Modèle de corners et de cartons par équipe.

Principe : pour chaque match, on estime le nombre attendu de corners
(ou de cartons) de l'équipe à domicile et de l'équipe à l'extérieur à
partir de la force "offensive" de chaque équipe, de la "résistance"
de l'adversaire, d'un avantage du terrain et d'un niveau général, avec
une décroissance du poids des vieux matchs. Le total est ensuite
modélisé par une loi binomiale négative (les totaux varient plus qu'une
loi de Poisson ne le prévoit).

Ce que ce module PEUT faire : mesurer si connaître les équipes aide à
prédire mieux que la simple moyenne du championnat.
Ce qu'il NE PEUT PAS faire : prouver qu'on bat les bookmakers. Les
fichiers gratuits ne contiennent pas les cotes de corners ni de cartons.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import nbinom, poisson

import dixon_coles as dc

KINDS = {
    "Corners": {
        "home": ["HC"],
        "away": ["AC"],
        "lines": [7.5, 8.5, 9.5, 10.5, 11.5, 12.5],
        "default": [8.5, 9.5, 10.5],
    },
    "Cartons": {
        # Un carton rouge compte pour 1 carton ici. Beaucoup de
        # bookmakers utilisent des "points" (jaune = 1, rouge = 2 ou 3) :
        # vérifie la règle du marché avant de comparer.
        "home": ["HY", "HR"],
        "away": ["AY", "AR"],
        "lines": [2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
        "default": [3.5, 4.5, 5.5],
    },
}

EMPTY = pd.DataFrame(columns=["Date", "HomeTeam", "AwayTeam", "h", "a"])


# ----------------------------------------------------------------------
# Données
# ----------------------------------------------------------------------
def load_counts(code, kind, n_seasons=5, cache="data"):
    """Charge, pour un championnat, les comptes domicile / extérieur.

    Retourne un DataFrame [Date, HomeTeam, AwayTeam, h, a] ou un
    DataFrame vide si les colonnes n'existent pas pour ce championnat.
    """
    cfg = KINDS[kind]
    needed = ["Date", "HomeTeam", "AwayTeam"] + cfg["home"] + cfg["away"]
    frames = []

    for k, season in enumerate(dc.season_codes(n_seasons)):
        raw = dc._load_csv(code, season, cache, refresh=(k == 0))
        if raw is None or not set(needed).issubset(raw.columns):
            continue

        d = raw[needed].copy()
        d["Date"] = pd.to_datetime(d["Date"], dayfirst=True, errors="coerce")
        for col in cfg["home"] + cfg["away"]:
            d[col] = pd.to_numeric(d[col], errors="coerce")

        # Une ligne n'est gardée que si TOUTES les valeurs existent :
        # on n'invente jamais de donnée manquante.
        d = d.dropna()
        if d.empty:
            continue

        d["h"] = d[cfg["home"]].sum(axis=1).astype(int)
        d["a"] = d[cfg["away"]].sum(axis=1).astype(int)
        frames.append(d[["Date", "HomeTeam", "AwayTeam", "h", "a"]])

    if not frames:
        return EMPTY.copy()

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(["Date", "HomeTeam", "AwayTeam"])
    return out.sort_values("Date").reset_index(drop=True)


# ----------------------------------------------------------------------
# Modèle
# ----------------------------------------------------------------------
class CountModel:
    """Poisson pondéré dans le temps + dispersion globale (binomiale négative)."""

    def __init__(self, xi=0.0018, ridge=30.0):
        self.xi = float(xi)
        self.ridge = float(ridge)

    def fit(self, df, ref_date=None):
        data = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "h", "a"])
        if len(data) < 100:
            raise ValueError("Moins de 100 matchs : historique insuffisant.")

        ref = pd.Timestamp(ref_date if ref_date is not None else data["Date"].max())
        teams = sorted(set(data["HomeTeam"]) | set(data["AwayTeam"]))
        index = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        hi = data["HomeTeam"].map(index).to_numpy()
        ai = data["AwayTeam"].map(index).to_numpy()
        h = data["h"].to_numpy(dtype=float)
        a = data["a"].to_numpy(dtype=float)
        days = (ref - data["Date"]).dt.days.clip(lower=0).to_numpy()
        w = np.exp(-self.xi * days)
        ridge = self.ridge

        def objective(p):
            att, de = p[:n], p[n:2 * n]
            ha, mu = p[2 * n], p[2 * n + 1]
            eh = np.clip(mu + att[hi] + de[ai] + ha, -5, 4)
            ea = np.clip(mu + att[ai] + de[hi], -5, 4)
            lh, la = np.exp(eh), np.exp(ea)

            nll = -(w * (h * eh - lh + a * ea - la)).sum()
            nll += ridge * (att @ att + de @ de)

            rh, ra = w * (h - lh), w * (a - la)
            g = np.zeros_like(p)
            g[:n] = -(np.bincount(hi, rh, n) + np.bincount(ai, ra, n)) + 2 * ridge * att
            g[n:2 * n] = -(np.bincount(ai, rh, n) + np.bincount(hi, ra, n)) + 2 * ridge * de
            g[2 * n] = -rh.sum()
            g[2 * n + 1] = -(rh.sum() + ra.sum())
            return nll, g

        x0 = np.zeros(2 * n + 2)
        x0[2 * n] = 0.05
        x0[2 * n + 1] = np.log(max((h.sum() + a.sum()) / (2 * len(h)), 0.5))

        res = minimize(objective, x0, jac=True, method="L-BFGS-B",
                       options={"maxiter": 500})
        if not np.isfinite(res.fun):
            raise RuntimeError(f"Échec du modèle : {res.message}")

        p = res.x
        self.teams, self.idx = teams, index
        self.att, self.de = p[:n].copy(), p[n:2 * n].copy()
        self.home_adv, self.mu = float(p[2 * n]), float(p[2 * n + 1])
        self.last_date = data["Date"].max()

        # Dispersion du total (statistique de Pearson pondérée).
        lh = np.exp(self.mu + self.att[hi] + self.de[ai] + self.home_adv)
        la = np.exp(self.mu + self.att[ai] + self.de[hi])
        m = lh + la
        y = h + a
        self.phi = float(max(1.0, (w * (y - m) ** 2 / m).sum() / w.sum()))

        # Équipes ayant joué récemment (pour ne pas proposer des clubs relégués).
        recent = data[data["Date"] >= self.last_date - pd.Timedelta(days=400)]
        self.active = sorted(set(recent["HomeTeam"]) | set(recent["AwayTeam"]))
        return self

    def rates(self, home, away):
        if home not in self.idx or away not in self.idx:
            raise KeyError("Équipe inconnue dans l'historique.")
        i, j = self.idx[home], self.idx[away]
        lh = np.exp(np.clip(self.mu + self.att[i] + self.de[j] + self.home_adv, -3, 3))
        la = np.exp(np.clip(self.mu + self.att[j] + self.de[i], -3, 3))
        return float(lh), float(la)

    def over_prob(self, mean_total, line):
        """P(total > line) pour une ligne en .5 (pas de remboursement possible)."""
        k = np.floor(line)
        if self.phi <= 1.0005:
            return float(poisson.sf(k, mean_total))
        r = mean_total / (self.phi - 1.0)
        return float(nbinom.sf(k, r, r / (r + mean_total)))

    def predict(self, home, away, lines):
        lh, la = self.rates(home, away)
        m = lh + la
        rows = []
        for line in lines:
            p_over = self.over_prob(m, line)
            rows.append({
                "ligne": line,
                "p_plus": p_over,
                "p_moins": 1.0 - p_over,
                "cote_juste_plus": 1.0 / p_over if p_over > 0 else np.inf,
                "cote_juste_moins": 1.0 / (1 - p_over) if p_over < 1 else np.inf,
            })
        return {"home": lh, "away": la, "total": m, "table": pd.DataFrame(rows)}


# ----------------------------------------------------------------------
# Backtest chronologique
# ----------------------------------------------------------------------
def _logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def backtest(df, lines, n_blocks=3, test_frac=0.30, xi=0.0018, ridge=30.0,
             min_train=300):
    """Compare le modèle par équipe à la simple fréquence du championnat.

    Pour chaque bloc de test, le modèle est ajusté uniquement sur les
    matchs strictement antérieurs. Retourne (détail, méta).
    """
    df = df.sort_values("Date").reset_index(drop=True)
    dates = np.sort(df["Date"].unique())
    if len(dates) < 20:
        return pd.DataFrame(), {}

    edges = np.linspace(int(len(dates) * (1 - test_frac)), len(dates),
                        n_blocks + 1).astype(int)
    records, mae_model, mae_base = [], [], []

    for b in range(n_blocks):
        lo_date = dates[min(edges[b], len(dates) - 1)]
        hi_date = dates[edges[b + 1]] if edges[b + 1] < len(dates) else None

        train = df[df["Date"] < lo_date]
        test = df[df["Date"] >= lo_date]
        if hi_date is not None:
            test = test[test["Date"] < hi_date]
        if len(train) < min_train or test.empty:
            continue

        model = CountModel(xi, ridge).fit(train, ref_date=train["Date"].max())
        base_total = float((train["h"] + train["a"]).mean())
        train_total = (train["h"] + train["a"]).to_numpy()

        known = test[test["HomeTeam"].isin(model.idx) & test["AwayTeam"].isin(model.idx)]
        for r in known.itertuples(index=False):
            lh, la = model.rates(r.HomeTeam, r.AwayTeam)
            m, total = lh + la, r.h + r.a
            mae_model.append(abs(m - total))
            mae_base.append(abs(base_total - total))
            for line in lines:
                y = float(total > line)
                p_base = ((train_total > line).sum() + 1) / (len(train_total) + 2)
                p_model = model.over_prob(m, line)
                records.append((line, y, p_model, p_base))

    if not records:
        return pd.DataFrame(), {}

    out = pd.DataFrame(records, columns=["ligne", "y", "p_modele", "p_base"])
    out["ll_modele"] = _logloss(out["p_modele"].to_numpy(), out["y"].to_numpy())
    out["ll_base"] = _logloss(out["p_base"].to_numpy(), out["y"].to_numpy())
    meta = {
        "n_matchs": len(out) // len(lines),
        "mae_modele": float(np.mean(mae_model)),
        "mae_base": float(np.mean(mae_base)),
    }
    return out, meta


def summarize_backtest(detail):
    """Une ligne par ligne de marché : log-loss modèle vs base, écart et z."""
    rows = []
    for line, g in detail.groupby("ligne"):
        diff = (g["ll_base"] - g["ll_modele"]).to_numpy()  # >0 : le modèle fait mieux
        n = len(diff)
        se = diff.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        rows.append({
            "ligne": line,
            "matchs": n,
            "perte_modele": g["ll_modele"].mean(),
            "perte_moyenne_ligue": g["ll_base"].mean(),
            "gain_moyen": diff.mean(),
            "erreur_std": se,
            "z": diff.mean() / se if se and np.isfinite(se) and se > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def calibration(detail, line, bins=5):
    """Probabilités annoncées (par tranches) vs fréquence réellement observée."""
    g = detail[detail["ligne"] == line]
    if len(g) < bins * 20:
        return pd.DataFrame()
    cut = pd.qcut(g["p_modele"], bins, duplicates="drop")
    t = g.groupby(cut, observed=True).agg(
        matchs=("y", "size"),
        annoncee=("p_modele", "mean"),
        observee=("y", "mean"),
    ).reset_index(drop=True)
    return t


# ----------------------------------------------------------------------
# Comparaison à une cote saisie
# ----------------------------------------------------------------------
def compare_to_odds(p_over, odds_over, odds_under):
    """Compare la probabilité du modèle à celle du bookmaker.

    Si les deux cotes sont données, la marge est retirée proportionnellement.
    Sinon la probabilité implicite brute (marge incluse) est utilisée.
    """
    out = {}
    if odds_over and odds_over > 1:
        if odds_under and odds_under > 1:
            io, iu = 1 / odds_over, 1 / odds_under
            imp = io / (io + iu)
            out["marge_pct"] = (io + iu - 1) * 100
        else:
            imp = 1 / odds_over
        out["implicite_plus"] = imp
        out["ecart_plus_pts"] = (p_over - imp) * 100
        out["ev_plus_pct"] = (p_over * odds_over - 1) * 100
    if odds_under and odds_under > 1:
        if odds_over and odds_over > 1:
            io, iu = 1 / odds_over, 1 / odds_under
            imp = iu / (io + iu)
        else:
            imp = 1 / odds_under
        out["implicite_moins"] = imp
        out["ecart_moins_pts"] = ((1 - p_over) - imp) * 100
        out["ev_moins_pct"] = ((1 - p_over) * odds_under - 1) * 100
    return out
