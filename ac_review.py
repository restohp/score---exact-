"""Auto-correction (3/3) : revue match par match, équipes mal évaluées, mi-temps."""
import numpy as np
import pandas as pd
from scipy.stats import poisson

from ac_data import HALF_SHARE, _with_half, half_scores, league_share
from ac_fit import _z


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
