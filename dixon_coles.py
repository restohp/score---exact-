import os, sys, time, datetime as dt
import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize
from scipy.stats import poisson

URL = "https://www.football-data.co.uk/mmz4281/{s}/{code}.csv"

LEAGUES = {
    "E0": "Premier League", "E1": "Championship",
    "E2": "League One", "E3": "League Two",
    "SP1": "La Liga", "SP2": "La Liga 2",
    "D1": "Bundesliga", "D2": "Bundesliga 2",
    "I1": "Serie A", "I2": "Serie B",
    "F1": "Ligue 1", "F2": "Ligue 2",
    "N1": "Eredivisie", "B1": "Jupiler Pro League",
    "P1": "Liga Portugal", "SC0": "Scottish Premiership",
    "T1": "Super Lig", "G1": "Super League Grece",
}

SHOT_COLS = ["HS", "AS", "HST", "AST"]
KEEP = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG",
        "B365H", "B365D", "B365A", "AvgH", "AvgD", "AvgA"] + SHOT_COLS


def season_codes(n=4):
    today = dt.date.today()
    start = today.year if today.month >= 8 else today.year - 1
    return [f"{y % 100:02d}{(y + 1) % 100:02d}"
            for y in range(start, start - n, -1)]


def load_league(code, n_seasons=4, cache="data"):
    os.makedirs(cache, exist_ok=True)
    frames = []
    for k, s in enumerate(season_codes(n_seasons)):
        path = f"{cache}/{code}_{s}.csv"
        stale = k == 0 and (not os.path.exists(path)
                            or time.time() - os.path.getmtime(path) > 43200)
        if stale or not os.path.exists(path):
            try:
                r = requests.get(URL.format(s=s, code=code), timeout=30)
                r.raise_for_status()
                with open(path, "wb") as f:
                    f.write(r.content)
            except Exception as e:
                print(f"  {code} {s}: telechargement impossible ({e})")
                if not os.path.exists(path):
                    continue
        try:
            d = pd.read_csv(path, encoding="latin-1")
        except Exception:
            continue
        cols = [c for c in KEEP if c in d.columns]
        d = d[cols].dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
        d["Date"] = pd.to_datetime(d["Date"], dayfirst=True, errors="coerce")
        d = d.dropna(subset=["Date"])
        for c in SHOT_COLS:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        d["League"] = code
        frames.append(d)
    if not frames:
        raise RuntimeError(f"Aucune donnee pour {code}")
    return pd.concat(frames).sort_values("Date").reset_index(drop=True)


class DixonColes:
    """att = force d'attaque, dfn = faiblesse defensive (plus haut = encaisse plus).

    xi       : decroissance par jour (demi-vie ~ 1 an a 0.0018)
    shots_w  : 0 = buts seuls, 1 = buts attendus estimes a partir des tirs,
               entre les deux = melange
    """

    def __init__(self, xi=0.0018, shots_w=0.0):
        self.xi = xi
        self.shots_w = shots_w

    @staticmethod
    def _proxy(df):
        """Buts attendus estimes depuis les tirs (regression sur le jeu d'entrainement)."""
        if not all(c in df.columns for c in SHOT_COLS):
            return None, None
        hs, as_, hst, ast = (pd.to_numeric(df[c], errors="coerce").values
                             for c in SHOT_COLS)
        ok = ~(np.isnan(hs) | np.isnan(as_) | np.isnan(hst) | np.isnan(ast))
        if ok.sum() < 100:
            return None, None
        X = np.concatenate([
            np.column_stack([hst[ok], (hs - hst)[ok]]),
            np.column_stack([ast[ok], (as_ - ast)[ok]]),
        ])
        y = np.concatenate([df.FTHG.values[ok].astype(float),
                            df.FTAG.values[ok].astype(float)])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        coef = np.clip(coef, 0, None)
        ph = coef[0] * hst + coef[1] * (hs - hst)
        pa = coef[0] * ast + coef[1] * (as_ - ast)
        return ph, pa

    def fit(self, df, ref_date=None):
        ref_date = ref_date or df["Date"].max()
        teams = sorted(set(df.HomeTeam) | set(df.AwayTeam))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        h = df.HomeTeam.map(idx).values
        a = df.AwayTeam.map(idx).values
        hg = df.FTHG.values.astype(int)
        ag = df.FTAG.values.astype(int)
        days = (ref_date - df["Date"]).dt.days.clip(lower=0).values
        w = np.exp(-self.xi * days)

        # cibles : buts reels, ou melange buts / buts estimes par les tirs
        th, ta = hg.astype(float), ag.astype(float)
        if self.shots_w > 0:
            ph, pa = self._proxy(df)
            if ph is not None:
                sw = self.shots_w
                th = np.where(np.isnan(ph), th, (1 - sw) * th + sw * ph)
                ta = np.where(np.isnan(pa), ta, (1 - sw) * ta + sw * pa)

        m00 = (hg == 0) & (ag == 0)
        m01 = (hg == 0) & (ag == 1)
        m10 = (hg == 1) & (ag == 0)
        m11 = (hg == 1) & (ag == 1)

        def unpack(p):
            att = p[:n] - p[:n].mean()
            return att, p[n:2 * n], p[2 * n], p[2 * n + 1]

        def nll(p):
            att, dfn, home, rho = unpack(p)
            lh = np.exp(att[h] + dfn[a] + home)
            la = np.exp(att[a] + dfn[h])
            tau = np.ones_like(lh)
            tau[m00] = 1 - lh[m00] * la[m00] * rho
            tau[m01] = 1 + lh[m01] * rho
            tau[m10] = 1 + la[m10] * rho
            tau[m11] = 1 - rho
            tau = np.clip(tau, 1e-6, None)
            ll = (np.log(tau) + th * np.log(lh) - lh
                  + ta * np.log(la) - la)
            return -(w * ll).sum()

        p0 = np.zeros(2 * n + 2)
        p0[2 * n] = 0.25
        p0[2 * n + 1] = -0.05
        bounds = [(None, None)] * (2 * n + 1) + [(-0.3, 0.3)]
        res = minimize(nll, p0, method="L-BFGS-B", bounds=bounds)
        self.teams, self.idx = teams, idx
        self.att, self.dfn, self.home, self.rho = unpack(res.x)
        return self

    def rates(self, home, away):
        i, j = self.idx[home], self.idx[away]
        lh = np.exp(self.att[i] + self.dfn[j] + self.home)
        la = np.exp(self.att[j] + self.dfn[i])
        return lh, la

    def score_matrix(self, home, away, max_goals=8):
        lh, la = self.rates(home, away)
        g = np.arange(max_goals + 1)
        M = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
        r = self.rho
        M[0, 0] *= max(1 - lh * la * r, 0)
        M[0, 1] *= 1 + lh * r
        M[1, 0] *= 1 + la * r
        M[1, 1] *= 1 - r
        return M / M.sum()

    def summary(self, home, away, top=5):
        M = self.score_matrix(home, away)
        lh, la = self.rates(home, away)
        n = M.shape[0]
        scores = sorted(((i, j, M[i, j]) for i in range(n) for j in range(n)),
                        key=lambda x: -x[2])[:top]
        over25 = sum(M[i, j] for i in range(n) for j in range(n) if i + j > 2)
        return {
            "buts_attendus": (round(lh, 2), round(la, 2)),
            "top_scores": [(f"{i}-{j}", round(p * 100, 1)) for i, j, p in scores],
            "1": round(np.tril(M, -1).sum() * 100, 1),
            "X": round(np.trace(M) * 100, 1),
            "2": round(np.triu(M, 1).sum() * 100, 1),
            "over_2.5": round(over25 * 100, 1),
            "btts": round((1 - M[0, :].sum() - M[:, 0].sum() + M[0, 0]) * 100, 1),
        }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Ligues disponibles :")
        for c, nom in LEAGUES.items():
            print(f"  {c:4} {nom}")
        print('\nUsage : python dixon_coles.py F1')
        print('        python dixon_coles.py F1 "Paris SG" "Lyon"')
        sys.exit()

    code = sys.argv[1].upper()
    df = load_league(code)
    print(f"{LEAGUES.get(code, code)} : {len(df)} matchs charges")
    model = DixonColes().fit(df)

    if len(sys.argv) < 4:
        recent = set(df[df.Date > df.Date.max() - pd.Timedelta(days=400)].HomeTeam)
        print("Equipes :", ", ".join(sorted(recent)))
        sys.exit()

    home, away = sys.argv[2], sys.argv[3]
    for t in (home, away):
        if t not in model.idx:
            sys.exit(f"Equipe inconnue : {t} (lance sans equipes pour voir la liste)")
    r = model.summary(home, away)
    print(f"\n{home} vs {away}")
    print("Buts attendus :", r["buts_attendus"])
    for s, p in r["top_scores"]:
        print(f"  {s} : {p}%")
    print(f"1: {r['1']}%  X: {r['X']}%  2: {r['2']}%")
    print(f"Over 2.5: {r['over_2.5']}%  BTTS: {r['btts']}%")
