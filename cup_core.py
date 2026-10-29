"""Coupes européennes : lecture des résultats openfootball + modèle de force des clubs.

Source : https://github.com/openfootball/champions-league (domaine public, CC0).
Le modèle est le même Dixon-Coles que pour les championnats, ajusté sur les
matchs européens des dernières saisons (c'est eux qui relient les pays entre eux).
"""
import re
from datetime import date

import pandas as pd
import requests

from dixon_coles import DixonColes

BASE = "https://raw.githubusercontent.com/openfootball/champions-league/master"
SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26", "2026-27"]
FILES = {"cl.txt": "Ligue des champions", "el.txt": "Ligue Europa",
         "conf.txt": "Ligue Conférence"}

MONTHS = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}
RE_DATE = re.compile(r"^\s{1,3}[A-Z][a-z]{2}\s+([A-Z][a-z]{2})\s+(\d{1,2})(?:\s+(\d{4}))?\s*$")
RE_HEADER_DATE = re.compile(r"^#\s*Date\s+[A-Z][a-z]{2}\s+[A-Z][a-z]{2}\s+\d{1,2}\s+(\d{4})")
RE_PLAYED = re.compile(
    r"^\s+(?:\d{1,2}:\d{2}\s+)?(?P<h>.+?)\s+v\s+(?P<a>.+?)\s+(?P<s1>\d+)-(?P<s2>\d+)(?P<rest>.*)$")
RE_FIXTURE = re.compile(r"^\s+(?:\d{1,2}:\d{2}\s+)?(?P<h>.+?)\s+v\s+(?P<a>[^\d]+?)\s*$")
RE_PEN = re.compile(r"^\s*pen\.\s+(\d+)-(\d+)")
RE_COUNTRY = re.compile(r"\s*\(([A-Z]{3})\)\s*$")


def _clean(name):
    return RE_COUNTRY.sub("", name).strip()


def parse_text(text, competition=""):
    """Texte openfootball -> (matchs joués, matchs à venir)."""
    year = None
    for line in text.splitlines()[:12]:
        m = RE_HEADER_DATE.match(line)
        if m:
            year = int(m.group(1))
            break
    cur, last_month = None, None
    played, fixtures = [], []
    for line in text.splitlines():
        d = RE_DATE.match(line)
        if d:
            month, day, y = MONTHS.get(d.group(1)), int(d.group(2)), d.group(3)
            if month is None:
                continue
            if y:
                year = int(y)
            elif year is not None and last_month is not None and month < last_month:
                year += 1
            last_month = month
            try:
                cur = date(year, month, day) if year else None
            except ValueError:
                cur = None
            continue
        if cur is None or line.lstrip().startswith(("#", "=", "▪")):
            continue
        m = RE_PLAYED.match(line)
        if m:
            rest = m.group("rest")
            p = RE_PEN.match(rest)
            # tirs au but : le score du match est donné juste après « pen. »
            if p:
                hg, ag = int(p.group(1)), int(p.group(2))
            else:
                hg, ag = int(m.group("s1")), int(m.group("s2"))
            played.append({"Date": pd.Timestamp(cur), "HomeTeam": _clean(m.group("h")),
                           "AwayTeam": _clean(m.group("a")), "FTHG": hg, "FTAG": ag,
                           "Competition": competition})
            continue
        f = RE_FIXTURE.match(line)
        if f and " v " in line:
            fixtures.append({"Date": pd.Timestamp(cur), "HomeTeam": _clean(f.group("h")),
                             "AwayTeam": _clean(f.group("a")), "Competition": competition})
    return played, fixtures


def _get(url):
    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 200 and len(r.text) > 200:
            return r.text
    except requests.RequestException:
        pass
    return None


def load_cups(seasons=None):
    """Télécharge les saisons et renvoie (df_joués, df_à_venir, rapport)."""
    played, fixtures, report = [], [], []
    for s in (seasons or SEASONS):
        for fname, label in FILES.items():
            txt = _get(f"{BASE}/{s}/{fname}")
            if txt is None:
                report.append((s, label, 0, "indisponible"))
                continue
            p, f = parse_text(txt, label)
            played += p
            fixtures += f
            report.append((s, label, len(p), "ok"))
    cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "Competition"]
    df = pd.DataFrame(played, columns=cols)
    if len(df):
        df = df.drop_duplicates(["Date", "HomeTeam", "AwayTeam"]).sort_values("Date").reset_index(drop=True)
    fx = pd.DataFrame(fixtures, columns=["Date", "HomeTeam", "AwayTeam", "Competition"])
    return df, fx, report


def fit_model(df, ref_date=None):
    """Ajuste le modèle sur toute l'histoire européenne chargée."""
    ref = pd.Timestamp(ref_date) if ref_date is not None else df["Date"].max()
    ref = max(ref, df["Date"].max())
    return DixonColes(xi=0.0018).fit(df, ref_date=ref)


def team_table(df, ref_date=None):
    """Équipes actives (match dans les 500 derniers jours) avec nombre de matchs."""
    ref = pd.Timestamp(ref_date) if ref_date is not None else df["Date"].max()
    ref = max(ref, df["Date"].max())
    recent = df[df["Date"] >= ref - pd.Timedelta(days=500)]
    counts = pd.concat([recent["HomeTeam"], recent["AwayTeam"]]).value_counts()
    return counts.sort_index()
