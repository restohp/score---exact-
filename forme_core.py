"""Forme, statistiques de jeu, confrontations directes et modèle combiné.

Tout est calculé uniquement avec des matchs STRICTEMENT antérieurs à la date
du match analysé (aucune fuite du futur).

Le « proxy xG » n'est PAS un vrai xG : il mélange les buts et les tirs cadrés
(buts attendus ≈ tirs cadrés × taux de conversion moyen de la ligue).
"""
import numpy as np
import pandas as pd
from scipy.stats import poisson

import dixon_coles as dc
from dixon_coles import DixonColes

DECAY = 0.85          # poids du match k = DECAY ** k (k = 0 : le plus récent)
PRIOR = 4.0           # « matchs fictifs » de la moyenne de ligue (stabilise les petits échantillons)
WINDOWS = (5, 10, 15)
EXTRA = ["HS", "AS", "HST", "AST", "AvgH", "AvgD", "AvgA", "B365H", "B365D", "B365A"]


# ----------------------------------------------------------------------
# Données
# ----------------------------------------------------------------------
def load_stats(code, n_seasons=5, cache="data"):
    frames = []
    for k, season in enumerate(dc.season_codes(n_seasons)):
        raw = dc._load_csv(code, season, cache, refresh=(k == 0))
        core = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
        if raw is None or not set(core).issubset(raw.columns):
            continue
        raw = raw.copy()
        for c in EXTRA:
            if c not in raw.columns:
                raw[c] = np.nan
        d = raw[core + EXTRA].copy()
        d["Date"] = pd.to_datetime(d["Date"], dayfirst=True, errors="coerce")
        for c in ["FTHG", "FTAG"] + EXTRA:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.dropna(subset=core)
        d["FTHG"] = d["FTHG"].astype(int)
        d["FTAG"] = d["FTAG"].astype(int)
        frames.append(d)
    if not frames:
        raise RuntimeError("Aucune donnée trouvée pour ce championnat.")
    out = pd.concat(frames, ignore_index=True).drop_duplicates(["Date", "HomeTeam", "AwayTeam"])
    return out.sort_values("Date").reset_index(drop=True)


def _wmean(x, w):
    m = ~np.isnan(x)
    if not m.any():
        return np.nan
    return float((w[m] * x[m]).sum() / w[m].sum())


class Logs:
    """Journal des matchs de chaque équipe, interrogeable « avant une date »."""

    def __init__(self, df):
        self.df = df.sort_values("Date").reset_index(drop=True)
        d = self.df
        self.dates = d["Date"].to_numpy(dtype="datetime64[ns]")
        self.hg = d["FTHG"].to_numpy(float)
        self.ag = d["FTAG"].to_numpy(float)
        self.hst = d["HST"].to_numpy(float)
        self.ast = d["AST"].to_numpy(float)
        home = pd.DataFrame({
            "team": d["HomeTeam"], "date": d["Date"], "home": 1,
            "gf": d["FTHG"], "ga": d["FTAG"], "sf": d["HST"], "sa": d["AST"]})
        away = pd.DataFrame({
            "team": d["AwayTeam"], "date": d["Date"], "home": 0,
            "gf": d["FTAG"], "ga": d["FTHG"], "sf": d["AST"], "sa": d["HST"]})
        long = pd.concat([home, away], ignore_index=True).sort_values(["team", "date"], kind="stable")
        self.t = {}
        for team, g in long.groupby("team"):
            self.t[team] = {
                "date": g["date"].to_numpy(dtype="datetime64[ns]"),
                "home": g["home"].to_numpy(),
                "gf": g["gf"].to_numpy(float), "ga": g["ga"].to_numpy(float),
                "sf": g["sf"].to_numpy(float), "sa": g["sa"].to_numpy(float),
            }

    # -- moyennes de la ligue sur les 400 jours précédents -----------------
    def league(self, before):
        b = np.datetime64(pd.Timestamp(before))
        i1 = np.searchsorted(self.dates, b, side="left")
        i0 = np.searchsorted(self.dates, b - np.timedelta64(400, "D"), side="left")
        if i1 - i0 < 60:
            return None
        hg, ag = self.hg[i0:i1], self.ag[i0:i1]
        hs, as_ = self.hst[i0:i1], self.ast[i0:i1]
        ok = ~np.isnan(hs) & ~np.isnan(as_)
        conv = None
        if ok.sum() >= 60 and (hs[ok].sum() + as_[ok].sum()) > 0:
            conv = float((hg[ok].sum() + ag[ok].sum()) / (hs[ok].sum() + as_[ok].sum()))
        return {"hg": float(hg.mean()), "ag": float(ag.mean()),
                "g": float((hg.mean() + ag.mean()) / 2), "conv": conv}

    # -- fenêtre de N derniers matchs avant une date ------------------------
    def _window(self, team, before, n, venue=None):
        L = self.t.get(team)
        if L is None:
            return None
        i = np.searchsorted(L["date"], np.datetime64(pd.Timestamp(before)), side="left")
        idx = np.arange(i)
        if venue is not None:
            idx = idx[L["home"][:i] == venue]
        idx = idx[-n:][::-1]
        if len(idx) == 0:
            return None
        return L, idx, DECAY ** np.arange(len(idx))

    def stats(self, team, before, n, venue=None):
        """Moyennes pondérées par la récence sur les n derniers matchs."""
        win = self._window(team, before, n, venue)
        if win is None:
            return None
        L, idx, w = win
        gf, ga = L["gf"][idx], L["ga"][idx]
        pts = np.where(gf > ga, 3.0, np.where(gf == ga, 1.0, 0.0))
        return {
            "matchs": len(idx),
            "points": _wmean(pts, w), "buts_pour": _wmean(gf, w), "buts_contre": _wmean(ga, w),
            "tc_pour": _wmean(L["sf"][idx], w), "tc_contre": _wmean(L["sa"][idx], w),
            "victoires": int((gf > ga).sum()), "nuls": int((gf == ga).sum()),
            "defaites": int((gf < ga).sum()),
        }

    def _rate(self, team, before, n, venue, lg):
        win = self._window(team, before, n, venue)
        if win is None:
            return lg["g"], lg["g"]
        L, idx, w = win
        gf, ga, sf, sa = L["gf"][idx], L["ga"][idx], L["sf"][idx], L["sa"][idx]
        if lg["conv"] is not None:      # proxy xG : moitié buts, moitié tirs cadrés
            gf = np.where(np.isnan(sf), gf, 0.5 * gf + 0.5 * lg["conv"] * sf)
            ga = np.where(np.isnan(sa), ga, 0.5 * ga + 0.5 * lg["conv"] * sa)
        sw = w.sum()
        att = (float((w * gf).sum()) + PRIOR * lg["g"]) / (sw + PRIOR)
        dfn = (float((w * ga).sum()) + PRIOR * lg["g"]) / (sw + PRIOR)
        return att, dfn

    def form_probs(self, home, away, before, n=10):
        """Probabilités 1/N/2 d'un modèle de Poisson basé sur la forme récente."""
        lg = self.league(before)
        if lg is None or home not in self.t or away not in self.t:
            return None
        ah = 0.5 * np.array(self._rate(home, before, n, 1, lg)) + 0.5 * np.array(self._rate(home, before, n, None, lg))
        aa = 0.5 * np.array(self._rate(away, before, n, 0, lg)) + 0.5 * np.array(self._rate(away, before, n, None, lg))
        g = lg["g"]
        lam_h = (lg["hg"] / g) * ah[0] * aa[1] / g
        lam_a = (lg["ag"] / g) * aa[0] * ah[1] / g
        k = np.arange(9)
        m = np.outer(poisson.pmf(k, lam_h), poisson.pmf(k, lam_a))
        m /= m.sum()
        return DixonColes.probs_1x2(m)

    def h2h(self, home, away, before, years=5, max_n=6):
        d = self.df
        b = pd.Timestamp(before)
        sel = d[(d["Date"] < b) & (d["Date"] >= b - pd.Timedelta(days=365 * years))
                & (((d["HomeTeam"] == home) & (d["AwayTeam"] == away))
                   | ((d["HomeTeam"] == away) & (d["AwayTeam"] == home)))]
        return sel.sort_values("Date", ascending=False).head(max_n)


# ----------------------------------------------------------------------
# Cotes
# ----------------------------------------------------------------------
def market_from_row(row):
    odds = dc.valid_odds(row)
    return dc.market_probabilities(odds) if odds else None


def fair_probs(odds):
    """Cotes 1/N/2 (n'importe quel nombre de décimales) -> probabilités sans marge, marge."""
    odds = np.asarray(odds, float)
    raw = 1.0 / odds
    return raw / raw.sum(), float(raw.sum() - 1.0)


# ----------------------------------------------------------------------
# Test sur l'historique
# ----------------------------------------------------------------------
def _fit_dc(df, cutoff):
    return DixonColes().fit(df[df["Date"] < cutoff], ref_date=cutoff - pd.Timedelta(days=1))


def backtest(df, n=10, test_frac=0.4, block_days=45, min_train=600):
    """Prédit chaque match test avec les seuls matchs antérieurs. Renvoie un dict de tableaux."""
    df = df.sort_values("Date").reset_index(drop=True)
    logs = Logs(df)
    start_idx = max(int(len(df) * (1 - test_frac)), min_train)
    if start_idx >= len(df) - 50:
        raise ValueError("Pas assez de matchs pour un test fiable.")
    t0 = df["Date"].iloc[start_idx]
    t_end = df["Date"].max()
    P_dc, P_fo, P_mk, Y, D = [], [], [], [], []
    block_start = t0
    while block_start <= t_end:
        block_end = block_start + pd.Timedelta(days=block_days)
        blk = df[(df["Date"] >= block_start) & (df["Date"] < block_end)]
        if len(blk):
            model = _fit_dc(df, block_start)
            for _, r in blk.iterrows():
                mk = market_from_row(r)
                if mk is None:
                    continue
                try:
                    p_dc = DixonColes.probs_1x2(model.score_matrix(r["HomeTeam"], r["AwayTeam"]))
                except Exception:
                    continue
                p_fo = logs.form_probs(r["HomeTeam"], r["AwayTeam"], r["Date"], n=n)
                if p_fo is None:
                    continue
                P_dc.append(p_dc); P_fo.append(p_fo); P_mk.append(mk)
                Y.append(dc.outcome_1x2(r["FTHG"], r["FTAG"])); D.append(r["Date"])
        block_start = block_end
    if len(Y) < 100:
        raise ValueError("Pas assez de matchs avec cotes pour un test fiable.")
    return {"dc": np.array(P_dc), "form": np.array(P_fo), "mkt": np.array(P_mk),
            "y": np.array(Y), "date": np.array(D)}


def _logloss(P, y):
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))


def _brier(P, y):
    Y = np.eye(3)[y]
    return ((P - Y) ** 2).sum(axis=1)


def summarize(bt, grid=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """Compare DC, forme, combinaison et marché. Le poids de la combinaison est choisi
    sur la 1re moitié du test et jugé sur la 2e moitié (aucune triche)."""
    y = bt["y"]
    n = len(y)
    h = n // 2
    first, second = np.arange(h), np.arange(h, n)
    ll = {w: _logloss(w * bt["dc"] + (1 - w) * bt["form"], y) for w in grid}
    best_w = min(grid, key=lambda w: ll[w][first].mean())
    ens = best_w * bt["dc"] + (1 - best_w) * bt["form"]
    models = {"Dixon-Coles": bt["dc"], "Forme récente": bt["form"],
              f"Combiné (poids DC {best_w:.2f})": ens, "Marché (sans marge)": bt["mkt"]}
    mk_ll = _logloss(bt["mkt"], y)
    rows = []
    for name, P in models.items():
        l = _logloss(P, y)
        d = (l - mk_ll)[second]
        z = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if name != "Marché (sans marge)" and d.std() > 0 else np.nan
        rows.append({"Modèle": name, "Log-loss": float(l.mean()), "Brier": float(_brier(P, y).mean()),
                     "Écart au marché (2e moitié)": float(d.mean()) if name != "Marché (sans marge)" else 0.0,
                     "z (positif = pire que le marché)": z})
    return pd.DataFrame(rows), best_w, n
