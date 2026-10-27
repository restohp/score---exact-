from __future__ import annotations

import datetime as dt
import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import requests

from scipy.optimize import minimize
from scipy.stats import poisson


URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"

LEAGUES = {
    "E0": "Premier League",
    "E1": "Championship",
    "E2": "League One",
    "E3": "League Two",
    "SP1": "La Liga",
    "SP2": "La Liga 2",
    "D1": "Bundesliga",
    "D2": "Bundesliga 2",
    "I1": "Serie A",
    "I2": "Serie B",
    "F1": "Ligue 1",
    "F2": "Ligue 2",
    "N1": "Eredivisie",
    "B1": "Jupiler Pro League",
    "P1": "Liga Portugal",
    "SC0": "Scottish Premiership",
    "T1": "Super Lig",
    "G1": "Super League Grece",
}

SHOT_COLS = ["HS", "AS", "HST", "AST"]

KEEP = [
    "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG",
    "B365H", "B365D", "B365A",
    "AvgH", "AvgD", "AvgA",
] + SHOT_COLS


def season_codes(n: int = 4):
    """Codes Football-Data, par exemple 2627 pour 2026-2027."""
    today = dt.date.today()
    start = today.year if today.month >= 8 else today.year - 1

    return [
        f"{y % 100:02d}{(y + 1) % 100:02d}"
        for y in range(start, start - n, -1)
    ]


def _read_csv(path_or_url):
    return pd.read_csv(
        path_or_url,
        encoding="latin-1",
        low_memory=False,
    )


def load_league(
    code: str,
    n_seasons: int = 4,
    cache: str = "data",
) -> pd.DataFrame:

    os.makedirs(cache, exist_ok=True)
    frames = []

    for k, season in enumerate(season_codes(n_seasons)):
        path = os.path.join(cache, f"{code}_{season}.csv")

        stale = k == 0 and (
            not os.path.exists(path)
            or time.time() - os.path.getmtime(path) > 43200
        )

        if stale or not os.path.exists(path):
            try:
                response = requests.get(
                    URL.format(season=season, code=code),
                    timeout=30,
                )
                response.raise_for_status()

                # Évite de sauvegarder une page HTML comme fichier CSV.
                if b"<html" in response.content[:500].lower():
                    raise ValueError(
                        "Le serveur a renvoyé une page HTML, pas un CSV."
                    )

                with open(path, "wb") as f:
                    f.write(response.content)

            except Exception as exc:
                print(
                    f"{code} {season}: "
                    f"téléchargement impossible: {exc}"
                )

                if not os.path.exists(path):
                    continue

        try:
            d = _read_csv(path)
        except Exception as exc:
            print(f"Lecture impossible {path}: {exc}")
            continue

        required = {
            "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"
        }

        if not required.issubset(d.columns):
            continue

        for col in KEEP:
            if col not in d.columns:
                d[col] = np.nan

        d = d[KEEP].copy()

        d["Date"] = pd.to_datetime(
            d["Date"],
            dayfirst=True,
            errors="coerce",
        )

        numeric_cols = [
            "FTHG", "FTAG",
            *SHOT_COLS,
            "B365H", "B365D", "B365A",
            "AvgH", "AvgD", "AvgA",
        ]

        for col in numeric_cols:
            d[col] = pd.to_numeric(
                d[col],
                errors="coerce",
            )

        d = d.dropna(
            subset=[
                "Date", "HomeTeam", "AwayTeam",
                "FTHG", "FTAG",
            ]
        )

        d["FTHG"] = d["FTHG"].astype(int)
        d["FTAG"] = d["FTAG"].astype(int)
        d["League"] = code

        frames.append(d)

    if not frames:
        raise RuntimeError(
            f"Aucune donnée historique trouvée pour {code}. "
            "Vérifie la connexion et le code du championnat."
        )

    out = pd.concat(frames, ignore_index=True)

    out = out.drop_duplicates(
        subset=["Date", "HomeTeam", "AwayTeam"],
        keep="last",
    )

    return out.sort_values("Date").reset_index(drop=True)


class DixonColes:
    """
    Modèle Dixon-Coles avec pondération temporelle.

    xi contrôle la décroissance de l'importance des matchs anciens.
    rho corrige les probabilités de certains scores faibles.
    """

    def __init__(
        self,
        xi: float = 0.0018,
        shots_w: float = 0.0,
    ):
        self.xi = float(xi)
        self.shots_w = float(shots_w)

    def fit(
        self,
        df: pd.DataFrame,
        ref_date: Optional[pd.Timestamp] = None,
    ):

        required = {
            "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"
        }

        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"Colonnes manquantes: {sorted(missing)}"
            )

        d = (
            df.dropna(subset=list(required))
            .copy()
            .sort_values("Date")
        )

        if len(d) < 30:
            raise ValueError(
                "Il faut au moins 30 matchs historiques."
            )

        ref_date = (
            pd.Timestamp(ref_date)
            if ref_date is not None
            else d["Date"].max()
        )

        teams = sorted(
            set(d.HomeTeam) | set(d.AwayTeam)
        )

        idx = {
            team: i
            for i, team in enumerate(teams)
        }

        h = d.HomeTeam.map(idx).to_numpy()
        a = d.AwayTeam.map(idx).to_numpy()

        hg = d.FTHG.to_numpy(dtype=int)
        ag = d.FTAG.to_numpy(dtype=int)

        days = (
            (ref_date - d.Date)
            .dt.days
            .clip(lower=0)
            .to_numpy()
        )

        weights = np.exp(-self.xi * days)
        n = len(teams)

        m00 = (hg == 0) & (ag == 0)
        m01 = (hg == 0) & (ag == 1)
        m10 = (hg == 1) & (ag == 0)
        m11 = (hg == 1) & (ag == 1)

        def unpack(p):
            # Centre les forces offensives pour éviter
            # une indétermination dans le niveau général.
            attack = p[:n] - p[:n].mean()
            defence = p[n:2*n]
            home_adv = p[2*n]
            rho = p[2*n + 1]

            return attack, defence, home_adv, rho

        def nll(p):
            attack, defence, home_adv, rho = unpack(p)

            lh = np.exp(
                np.clip(
                    attack[h] + defence[a] + home_adv,
                    -5,
                    4,
                )
            )

            la = np.exp(
                np.clip(
                    attack[a] + defence[h],
                    -5,
                    4,
                )
            )

            tau = np.ones(len(d), dtype=float)

            tau[m00] = (
                1 - lh[m00] * la[m00] * rho
            )

            tau[m01] = (
                1 + lh[m01] * rho
            )

            tau[m10] = (
                1 + la[m10] * rho
            )

            tau[m11] = 1 - rho

            tau = np.clip(tau, 1e-8, None)

            # Les termes de factorielle sont constants
            # par rapport aux paramètres et peuvent être omis.
            ll = (
                np.log(tau)
                + hg * np.log(lh) - lh
                + ag * np.log(la) - la
            )

            return -float(np.sum(weights * ll))

        p0 = np.zeros(2*n + 2, dtype=float)

        p0[2*n] = 0.20
        p0[2*n + 1] = -0.05

        bounds = (
            [(None, None)] * (2*n + 1)
            + [(-0.15, 0.15)]
        )

        result = minimize(
            nll,
            p0,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": 1500,
                "ftol": 1e-9,
            },
        )

        if not result.success and not np.isfinite(result.fun):
            raise RuntimeError(
                f"Échec de l'optimisation: {result.message}"
            )

        self.teams = teams
        self.idx = idx

        (
            self.attack,
            self.defence,
            self.home_adv,
            self.rho,
        ) = unpack(result.x)

        self.fit_date = ref_date
        self.optimization_ok = bool(result.success)

        return self

    def rates(self, home: str, away: str):
        """Retourne les buts attendus domicile et extérieur."""

        if home not in self.idx or away not in self.idx:
            raise KeyError(
                "Une équipe est absente de l'historique."
            )

        i = self.idx[home]
        j = self.idx[away]

        lh = float(
            np.exp(
                np.clip(
                    self.attack[i]
                    + self.defence[j]
                    + self.home_adv,
                    -5,
                    4,
                )
            )
        )

        la = float(
            np.exp(
                np.clip(
                    self.attack[j]
                    + self.defence[i],
                    -5,
                    4,
                )
            )
        )

        return lh, la

    def score_matrix(
        self,
        home: str,
        away: str,
        max_goals: int = 8,
    ):
        """Calcule les probabilités de chaque score exact."""

        lh, la = self.rates(home, away)

        goals = np.arange(max_goals + 1)

        matrix = np.outer(
            poisson.pmf(goals, lh),
            poisson.pmf(goals, la),
        )

        r = self.rho

        matrix[0, 0] *= max(
            1 - lh * la * r,
            1e-8,
        )

        matrix[0, 1] *= max(
            1 + lh * r,
            1e-8,
        )

        matrix[1, 0] *= max(
            1 + la * r,
            1e-8,
        )

        matrix[1, 1] *= max(
            1 - r,
            1e-8,
        )

        total = matrix.sum()

        if not np.isfinite(total) or total <= 0:
            raise RuntimeError(
                "La matrice des scores est invalide."
            )

        return matrix / total

    @staticmethod
    def probs_1x2(matrix):
        """Probabilités victoire domicile, nul, victoire extérieur."""

        return np.array([
            np.tril(matrix, -1).sum(),
            np.trace(matrix),
            np.triu(matrix, 1).sum(),
        ])

    def summary(
        self,
        home: str,
        away: str,
        top: int = 5,
    ):
        """Résumé statistique d'un match."""

        matrix = self.score_matrix(home, away)
        lh, la = self.rates(home, away)

        scores = sorted(
            (
                (i, j, matrix[i, j])
                for i in range(matrix.shape[0])
                for j in range(matrix.shape[1])
            ),
            key=lambda item: item[2],
            reverse=True,
        )[:top]

        over25 = sum(
            matrix[i, j]
            for i in range(matrix.shape[0])
            for j in range(matrix.shape[1])
            if i + j > 2
        )

        btts = (
            1
            - matrix[0, :].sum()
            - matrix[:, 0].sum()
            + matrix[0, 0]
        )

        p = self.probs_1x2(matrix)

        return {
            "buts_attendus": (
                round(lh, 2),
                round(la, 2),
            ),
            "top_scores": [
                (
                    f"{i}-{j}",
                    round(prob * 100, 2),
                )
                for i, j, prob in scores
            ],
            "1": round(p[0] * 100, 2),
            "X": round(p[1] * 100, 2),
            "2": round(p[2] * 100, 2),
            "over_2.5": round(over25 * 100, 2),
            "btts": round(btts * 100, 2),
        }
