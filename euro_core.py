"""Coupes d'Europe « pro » : force relative par l'Elo des clubs + style de jeu par la ligue d'origine.

- Elo (clubelo.com) : compare des clubs de pays différents sur une même échelle.
- Ligue d'origine (football-data.co.uk) : buts marqués / encaissés récents du club,
  qui donnent le rythme du match (beaucoup ou peu de buts).
- Les deux sont fusionnés : l'Elo décide QUI est favori, la ligue décide COMBIEN de buts.
Ce modèle ne peut pas être validé sur des matchs de coupe passés : estimation, pas test.
"""
import difflib
import io
import logging
import os
import re
import unicodedata

import numpy as np
import pandas as pd
import requests
from scipy.optimize import brentq
from scipy.stats import poisson

log = logging.getLogger(__name__)

FD_EXTRA = "https://www.football-data.co.uk/new/{c}.csv"

# pays ClubElo -> championnat football-data principal (déjà dans l'appli)
COUNTRY_MAIN = {"ENG": "E0", "ESP": "SP1", "GER": "D1", "ITA": "I1", "FRA": "F1", "NED": "N1",
                "BEL": "B1", "POR": "P1", "SCO": "SC0", "TUR": "T1", "GRE": "G1"}
# pays ClubElo -> fichier « autres championnats » football-data
COUNTRY_EXTRA = {"SWE": "SWE", "NOR": "NOR", "DEN": "DNK", "AUT": "AUT", "SUI": "SWZ",
                 "POL": "POL", "ROU": "ROU", "RUS": "RUS", "IRL": "IRL", "FIN": "FIN"}

BASE_TOTAL = 2.75      # buts moyens par match de coupe (valeur de départ, non validée)
HOME_ELO = 65.0        # avantage du terrain en points Elo
RHO = -0.06            # correction des petits scores (Dixon-Coles)
SHRINK = 10.0          # nombre de matchs « neutres » ajoutés pour stabiliser le style de jeu
MAXG = 10

ELO_TIMEOUT = (10, 30)  # (connexion, lecture) en secondes
UA = {"User-Agent": "Mozilla/5.0"}


# ---------------------------------------------------------------------------
# Elo
# ---------------------------------------------------------------------------
def _parse_elo(text):
    df = pd.read_csv(io.StringIO(text))
    df = df.dropna(subset=["Club", "Elo"])
    df["Elo"] = pd.to_numeric(df["Elo"], errors="coerce")
    df["Level"] = pd.to_numeric(df["Level"], errors="coerce")
    df = df.dropna(subset=["Elo"]).sort_values("Elo", ascending=False).reset_index(drop=True)
    if df.empty:
        raise ValueError("classement vide")
    return df


def fetch_elo(day=None, cache="data"):
    """Classement Elo de tous les clubs européens (DataFrame Club, Country, Level, Elo).

    1. Essaie clubelo.com (https puis http, aujourd'hui puis les 2 jours d'avant).
    2. Si le site est injoignable, utilise la copie data/clubelo.csv (celle du dépôt
       ou celle téléchargée la dernière fois).
    df.attrs["source"] indique d'où viennent les données.
    """
    os.makedirs(cache, exist_ok=True)
    path = f"{cache}/clubelo.csv"
    base = pd.Timestamp(day or pd.Timestamp.today())
    last = None
    unreachable = False

    for back in (0, 1, 2):
        d = (base - pd.Timedelta(days=back)).strftime("%Y-%m-%d")
        for scheme in ("https", "http"):
            try:
                r = requests.get(f"{scheme}://api.clubelo.com/{d}",
                                 timeout=ELO_TIMEOUT, headers=UA)
                r.raise_for_status()
                df = _parse_elo(r.text)
                try:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(r.text)
                except OSError:
                    pass  # disque en lecture seule : sans importance
                df.attrs["source"] = f"clubelo.com ({d})"
                return df
            except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError) as exc:
                last = exc
                unreachable = True
            except Exception as exc:
                last = exc
        if unreachable:
            break  # site injoignable : inutile d'essayer d'autres dates

    if os.path.exists(path):
        df = _parse_elo(open(path, encoding="utf-8").read())
        age = pd.Timestamp.today() - pd.Timestamp(os.path.getmtime(path), unit="s")
        df.attrs["source"] = f"copie locale (vieille de {age.days} j)"
        log.warning("clubelo.com injoignable (%s) : copie locale utilisée", type(last).__name__)
        return df

    raise RuntimeError(f"clubelo.com ne répond pas ({type(last).__name__}) "
                       f"et aucune copie locale dans {path}")


# ---------------------------------------------------------------------------
# Rapprochement des noms de clubs
# ---------------------------------------------------------------------------
_STOP = {"fc", "sc", "fk", "sk", "cf", "ac", "as", "ssc", "afc", "bk", "if", "ff",
         "sv", "vfb", "vfl", "rb", "ud", "cd", "sd", "cfr", "ca"}


def _norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(w for w in s.split() if w not in _STOP)


def match_team(name, candidates):
    """Nom ClubElo -> nom exact dans le fichier de la ligue, ou None si introuvable."""
    cands = {}
    for c in candidates:
        cands.setdefault(_norm(c), c)
    n = _norm(name)
    if not n:
        return None
    if n in cands:
        return cands[n]

    # inclusion par mots entiers (« kairat » dans « kairat almaty »), un seul candidat possible
    nw = set(n.split())
    hits = [v for k, v in cands.items()
            if k and (nw <= set(k.split()) or set(k.split()) <= nw)
            and min(len(n), len(k)) >= 4]
    if len(hits) == 1:
        return hits[0]

    best = difflib.get_close_matches(n, list(cands), n=1, cutoff=0.8)
    return cands[best[0]] if best else None


# ---------------------------------------------------------------------------
# Style de jeu par la ligue d'origine
# ---------------------------------------------------------------------------
def fetch_league_games(country):
    """Matchs récents (1 an) du championnat d'un pays : DataFrame Date, home, away, hg, ag. None si indisponible."""
    try:
        if country in COUNTRY_MAIN:
            import dixon_coles as dc
            d = dc.load_league(COUNTRY_MAIN[country], n_seasons=2)
            g = d.rename(columns={"HomeTeam": "home", "AwayTeam": "away", "FTHG": "hg", "FTAG": "ag"})
        elif country in COUNTRY_EXTRA:
            r = requests.get(FD_EXTRA.format(c=COUNTRY_EXTRA[country]), timeout=30, headers=UA)
            r.raise_for_status()
            raw = pd.read_csv(io.StringIO(r.content.decode("utf-8-sig", errors="replace")))
            raw.columns = [str(c).strip() for c in raw.columns]
            g = raw.rename(columns={"Home": "home", "Away": "away", "HG": "hg", "AG": "ag"})
            g["Date"] = pd.to_datetime(g["Date"], dayfirst=True, errors="coerce")
        else:
            return None
        g = g[["Date", "home", "away", "hg", "ag"]].copy()
        g["hg"] = pd.to_numeric(g["hg"], errors="coerce")
        g["ag"] = pd.to_numeric(g["ag"], errors="coerce")
        g = g.dropna()
        g = g[g["Date"] >= g["Date"].max() - pd.Timedelta(days=365)]
        return g.reset_index(drop=True) if len(g) >= 30 else None
    except Exception as exc:
        log.warning("ligue %s indisponible : %s: %s", country, type(exc).__name__, exc)
        return None


def team_style(games, team_in_data):
    """Rythme du club dans sa ligue : (buts marqués + encaissés) / 2 / moyenne de la ligue, stabilisé."""
    avg = (games["hg"].sum() + games["ag"].sum()) / (2 * len(games))
    h = games[games["home"] == team_in_data]
    a = games[games["away"] == team_in_data]
    n = len(h) + len(a)
    if n == 0 or avg <= 0:
        return 1.0, 0, avg
    gf = h["hg"].sum() + a["ag"].sum()
    ga = h["ag"].sum() + a["hg"].sum()
    raw = (gf + ga) / (2 * n) / avg
    return float((n * raw + SHRINK * 1.0) / (n + SHRINK)), int(n), float(avg)


# ---------------------------------------------------------------------------
# Modèle de scores
# ---------------------------------------------------------------------------
def _matrix(lh, la):
    ph, pa = poisson.pmf(np.arange(MAXG + 1), lh), poisson.pmf(np.arange(MAXG + 1), la)
    m = np.outer(ph, pa)
    m[0, 0] *= 1 - lh * la * RHO
    m[0, 1] *= 1 + lh * RHO
    m[1, 0] *= 1 + la * RHO
    m[1, 1] *= 1 - RHO
    return m / m.sum()


def _score_expect(m):
    i, j = np.indices(m.shape)
    return float(m[i > j].sum() + 0.5 * m[i == j].sum())


def elo_expect(elo_h, elo_a, neutral=False):
    d = elo_h - elo_a + (0.0 if neutral else HOME_ELO)
    return 1.0 / (1.0 + 10 ** (-d / 400.0))


def build_matrix(elo_h, elo_a, style_h, style_a, neutral=False):
    """Matrice de scores : total de buts par le style des deux clubs, répartition par l'Elo."""
    total = float(np.clip(BASE_TOTAL * (style_h + style_a) / 2.0, 0.75 * BASE_TOTAL, 1.3 * BASE_TOTAL))
    e = elo_expect(elo_h, elo_a, neutral)

    def gap(s):
        return _score_expect(_matrix(total * s, total * (1 - s))) - e

    s = brentq(gap, 0.03, 0.97)
    return _matrix(total * s, total * (1 - s)), total * s, total * (1 - s)


class MatrixModel:
    """Adaptateur : permet d'utiliser cartes_core.build_match avec une matrice déjà calculée."""

    def __init__(self, m):
        self.m = m

    def score_matrix(self, home, away):
        return self.m


def analyse(elo_df, home, away, neutral, league_cache):
    """Analyse d'un match. league_cache : dict pays -> DataFrame (ou None), rempli au besoin."""
    rows = {}
    for side, name in (("home", home), ("away", away)):
        r = elo_df[elo_df["Club"] == name].iloc[0]
        country = str(r["Country"])
        if country not in league_cache:
            league_cache[country] = fetch_league_games(country)
        games = league_cache[country]
        style, n, src = 1.0, 0, "Elo seul (ligue non disponible)"
        if games is not None:
            nm = match_team(name, set(games["home"]) | set(games["away"]))
            if nm is not None:
                style, n, avg = team_style(games, nm)
                src = f"ligue d'origine ({country}) : {n} matchs sur 1 an, style {style:.2f}"
            else:
                src = f"Elo seul (club introuvable dans la ligue {country})"
        rows[side] = {"club": name, "country": country, "elo": float(r["Elo"]),
                      "style": style, "n": n, "source": src}
    m, lh, la = build_matrix(rows["home"]["elo"], rows["away"]["elo"],
                             rows["home"]["style"], rows["away"]["style"], neutral)
    return {"matrix": m, "lh": lh, "la": la, "teams": rows}
