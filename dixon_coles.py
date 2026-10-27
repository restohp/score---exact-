import datetime as dt
import os
import time

import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize
from scipy.stats import poisson

URL = "https://www.football-data.co.uk/mmz4281/{s}/{code}.csv"

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

KEEP = [
    "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "AvgH", "AvgD", "AvgA",
    "B365H", "B365D", "B365A",
]


def season_codes(n=5):
    today = dt.date.today()
    start = today.year if today.month >= 8 else today.year - 1

    return [
        f"{y % 100:02d}{(y + 1) % 100:02d}"
        for y in range(start, start - n, -1)
    ]


def _load_csv(code, season, cache="data", refresh=False, refresh_hours=12):
    os.makedirs(cache, exist_ok=True)

    path = os.path.join(cache, f"{code}_{season}.csv")
    stale = (
        not os.path.exists(path)
        or (
            refresh
            and time.time() - os.path.getmtime(path) > refresh_hours * 3600
        )
    )

    if stale:
        try:
            response = requests.get(
                URL.format(s=season, code=code),
                timeout=25,
                headers={"User-Agent": "ScoreExactPro/1.0"},
            )
            response.raise_for_status()

            content = response.content
            if b"<html" not in content[:500].lower() and len(content) > 100:
                with open(path, "wb") as file:
                    file.write(content)

        except requests.RequestException as error:
            if not os.path.exists(path):
                print(f"Échec {code}/{season}: {error}")

    if not os.path.exists(path):
        return None

    try:
        return pd.read_csv(path, encoding="latin-1", low_memory=False)
    except Exception:
        return None


def load_league(code, n_seasons=5, cache="data"):
    if code not in LEAGUES:
        raise ValueError(f"Championnat inconnu : {code}")

    frames = []

    for k, season in enumerate(season_codes(n_seasons)):
        # Seule la saison courante est rafraîchie régulièrement.
        raw = _load_csv(code, season, cache, refresh=(k == 0))

        if raw is None:
            continue

        required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
        if not required.issubset(raw.columns):
            continue

        for column in KEEP:
            if column not in raw.columns:
                raw[column] = np.nan

        data = raw[KEEP].copy()
        data["Date"] = pd.to_datetime(
            data["Date"], dayfirst=True, errors="coerce"
        )

        for column in KEEP[3:]:
            data[column] = pd.to_numeric(
                data[column], errors="coerce"
            )

        data = data.dropna(
            subset=["Date", "HomeTeam", "AwayTeam"]
        )
        data["League"] = code
        frames.append(data)

    if not frames:
        raise RuntimeError(
            "Aucune donnée trouvée. Vérifie Internet et le championnat."
        )

    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(
        ["Date", "HomeTeam", "AwayTeam"]
    )

    return result.sort_values("Date").reset_index(drop=True)


def valid_odds(row):
    """Retourne les cotes 1/N/2 si elles sont valides."""

    for columns in [
        ("AvgH", "AvgD", "AvgA"),
        ("B365H", "B365D", "B365A"),
    ]:
        values = pd.to_numeric(
            pd.Series([row.get(c, np.nan) for c in columns]),
            errors="coerce",
        ).to_numpy(dtype=float)

        if np.isfinite(values).all() and (values > 1).all():
            return tuple(map(float, values))

    return None


def market_probabilities(odds):
    """Normalisation simple des probabilités implicites."""

    odds = np.asarray(odds, dtype=float)

    if odds.shape != (3,) or not np.isfinite(odds).all():
        raise ValueError("Trois cotes valides sont nécessaires.")

    if (odds <= 1).any():
        raise ValueError("Chaque cote doit être supérieure à 1.")

    probabilities = 1.0 / odds
    return probabilities / probabilities.sum()


def outcome_1x2(home_goals, away_goals):
    if home_goals > away_goals:
        return 0
    if home_goals == away_goals:
        return 1
    return 2


def probs_1x2(matrix):
    return np.array([
        np.tril(matrix, -1).sum(),
        np.trace(matrix),
        np.triu(matrix, 1).sum(),
    ])


class DixonColes:
    def __init__(self, xi=0.0018):
        self.xi = float(xi)

    def fit(self, df, ref_date=None):
        data = df.dropna(
            subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
        ).copy()

        if len(data) < 50:
            raise ValueError(
                "Moins de 50 matchs : historique insuffisant."
            )

        data["FTHG"] = data["FTHG"].astype(int)
        data["FTAG"] = data["FTAG"].astype(int)

        ref_date = pd.Timestamp(
            ref_date if ref_date is not None else data["Date"].max()
        )

        teams = sorted(set(data["HomeTeam"]) | set(data["AwayTeam"]))
        index = {team: i for i, team in enumerate(teams)}
        n = len(teams)

        home_idx = data["HomeTeam"].map(index).to_numpy()
        away_idx = data["AwayTeam"].map(index).to_numpy()

        home_goals = data["FTHG"].to_numpy()
        away_goals = data["FTAG"].to_numpy()

        days = (
            (ref_date - data["Date"]).dt.days.clip(lower=0).to_numpy()
        )
        weights = np.exp(-self.xi * days)

        m00 = (home_goals == 0) & (away_goals == 0)
        m01 = (home_goals == 0) & (away_goals == 1)
        m10 = (home_goals == 1) & (away_goals == 0)
        m11 = (home_goals == 1) & (away_goals == 1)

        def unpack(parameters):
            attack = parameters[:n] - parameters[:n].mean()
            defense = (
                parameters[n:2 * n] - parameters[n:2 * n].mean()
            )
            home_advantage = parameters[2 * n]
            rho = parameters[2 * n + 1]
            mu = parameters[2 * n + 2]  # niveau général des buts

            return attack, defense, home_advantage, rho, mu

        def negative_log_likelihood(parameters):
            attack, defense, home_advantage, rho, mu = unpack(parameters)

            lam_home = np.exp(np.clip(
                mu + attack[home_idx] + defense[away_idx] + home_advantage,
                -5, 4,
            ))
            lam_away = np.exp(np.clip(
                mu + attack[away_idx] + defense[home_idx],
                -5, 4,
            ))

            tau = np.ones(len(data))
            tau[m00] = 1 - lam_home[m00] * lam_away[m00] * rho
            tau[m01] = 1 + lam_home[m01] * rho
            tau[m10] = 1 + lam_away[m10] * rho
            tau[m11] = 1 - rho
            tau = np.clip(tau, 1e-8, None)

            # Les factorielles sont constantes : inutiles pour l'optimisation.
            likelihood = (
                np.log(tau)
                + home_goals * np.log(lam_home) - lam_home
                + away_goals * np.log(lam_away) - lam_away
            )

            penalty = 0.001 * np.square(parameters[:2 * n]).sum()

            return -float(np.sum(weights * likelihood)) + penalty

        initial = np.zeros(2 * n + 3)
        initial[2 * n] = 0.2
        initial[2 * n + 1] = -0.05
        initial[2 * n + 2] = 0.25

        bounds = (
            [(None, None)] * (2 * n + 1)
            + [(-0.25, 0.25)]
            + [(None, None)]
        )

        result = minimize(
            negative_log_likelihood,
            initial,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000},
        )

        if not np.isfinite(result.fun):
            raise RuntimeError(
                f"Échec du modèle : {result.message}"
            )

        self.teams = teams
        self.idx = index
        (
            self.att,
            self.defense,
            self.home,
            self.rho,
            self.mu,
        ) = unpack(result.x)
        self.optimization_message = str(result.message)

        return self

    def rates(self, home, away):
        if home not in self.idx or away not in self.idx:
            raise KeyError("Équipe inconnue dans l'historique.")

        i = self.idx[home]
        j = self.idx[away]

        home_rate = np.exp(np.clip(
            self.mu + self.att[i] + self.defense[j] + self.home,
            -3, 3,
        ))
        away_rate = np.exp(np.clip(
            self.mu + self.att[j] + self.defense[i],
            -3, 3,
        ))

        return float(home_rate), float(away_rate)

    def score_matrix(self, home, away, max_goals=8):
        home_rate, away_rate = self.rates(home, away)
        goals = np.arange(max_goals + 1)

        matrix = np.outer(
            poisson.pmf(goals, home_rate),
            poisson.pmf(goals, away_rate),
        )

        rho = self.rho
        matrix[0, 0] *= max(
            1 - home_rate * away_rate * rho, 1e-8
        )
        matrix[0, 1] *= max(1 + home_rate * rho, 1e-8)
        matrix[1, 0] *= max(1 + away_rate * rho, 1e-8)
        matrix[1, 1] *= max(1 - rho, 1e-8)

        total = matrix.sum()
        if total <= 0:
            raise RuntimeError("Matrice de scores invalide.")

        return matrix / total

    def summary(self, home, away, top=5):
        matrix = self.score_matrix(home, away)
        home_rate, away_rate = self.rates(home, away)

        cells = sorted(
            [
                (i, j, matrix[i, j])
                for i in range(matrix.shape[0])
                for j in range(matrix.shape[1])
            ],
            key=lambda item: -item[2],
        )[:top]

        over25 = sum(
            matrix[i, j]
            for i in range(matrix.shape[0])
            for j in range(matrix.shape[1])
            if i + j >= 3
        )

        btts = (
            1
            - matrix[0, :].sum()
            - matrix[:, 0].sum()
            + matrix[0, 0]
        )

        return {
            "matrix": matrix,
            "home_goals": home_rate,
            "away_goals": away_rate,
            "top_scores": [
                (f"{i}-{j}", float(probability * 100))
                for i, j, probability in cells
            ],
            "p_1x2": probs_1x2(matrix),
            "over25": float(over25 * 100),
            "btts": float(btts * 100),
        }
