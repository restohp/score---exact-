"""Simulation de matchs (Monte Carlo) à partir de la matrice de scores du modèle."""
import numpy as np
import pandas as pd

LINES = (0.5, 1.5, 2.5, 3.5, 4.5)
RES = ("Victoire domicile", "Match nul", "Victoire extérieur")


def simulate(matrix, n=20000, share=0.44, seed=None):
    """Tire n matchs au hasard. Chaque but tombe en 1re mi-temps avec probabilité `share`."""
    rng = np.random.default_rng(seed)
    m = np.asarray(matrix, float)
    m = m / m.sum()
    k = m.shape[1]
    n = int(n)
    cells = rng.choice(m.size, size=n, p=m.ravel())
    hg, ag = cells // k, cells % k
    h1 = rng.binomial(hg, share)
    a1 = rng.binomial(ag, share)
    return {"hg": hg, "ag": ag, "h1": h1, "a1": a1, "n": n}


def _res(h, a):
    return np.where(h > a, 0, np.where(h == a, 1, 2))


def exact_probs(matrix):
    """Probabilités exactes lues dans la matrice (sans tirage)."""
    m = np.asarray(matrix, float)
    m = m / m.sum()
    i, j = np.indices(m.shape)
    out = {"p1": float(m[i > j].sum()), "pn": float(m[i == j].sum()), "p2": float(m[i < j].sum())}
    tot = i + j
    for line in LINES:
        out[f"o{line}"] = float(m[tot > line].sum())
    out["gg"] = float(m[(i > 0) & (j > 0)].sum())
    out["lh"] = float((m * i).sum())
    out["la"] = float((m * j).sum())
    return out


def _ci(p, n):
    return 1.96 * np.sqrt(max(p * (1 - p), 0.0) / n)


def summarize(sim, matrix):
    """Tableaux prêts à afficher. Renvoie un dict de DataFrames et de valeurs."""
    hg, ag, h1, a1, n = sim["hg"], sim["ag"], sim["h1"], sim["a1"], sim["n"]
    ex = exact_probs(matrix)
    r = _res(hg, ag)
    sp = np.bincount(r, minlength=3) / n

    rows = []
    for k, (name, key) in enumerate(zip(RES, ("p1", "pn", "p2"))):
        rows.append((name, sp[k], ex[key], _ci(sp[k], n)))
    tot = hg + ag
    for line in LINES:
        p = float((tot > line).mean())
        rows.append((f"Plus de {line}".replace(".", ",") + " buts", p, ex[f"o{line}"], _ci(p, n)))
    gg = float(((hg > 0) & (ag > 0)).mean())
    rows.append(("Les deux marquent", gg, ex["gg"], _ci(gg, n)))
    bets = pd.DataFrame(rows, columns=["Pari", "Simulé", "Exact", "Marge ±"])

    key = hg * 20 + ag
    vals, cnt = np.unique(key, return_counts=True)
    order = np.argsort(-cnt)[:10]
    m = np.asarray(matrix, float)
    m = m / m.sum()
    top = pd.DataFrame({
        "Score": [f"{v // 20}-{v % 20}" for v in vals[order]],
        "Simulé": cnt[order] / n,
        "Exact": [float(m[v // 20, v % 20]) for v in vals[order]],
    })

    gcount = np.bincount(np.minimum(tot, 7), minlength=8) / n
    goals = pd.DataFrame({"Buts dans le match": [str(i) if i < 7 else "7+" for i in range(8)],
                          "Part des matchs": gcount}).set_index("Buts dans le match")

    ht = _res(h1, a1)
    htft = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            htft[a, b] = float(((ht == a) & (r == b)).mean())
    htft = pd.DataFrame(htft, index=[f"Mi-temps : {x.lower()}" for x in RES],
                        columns=[f"Fin : {x.lower()}" for x in RES])

    g1, g2 = h1 + a1, tot - h1 - a1
    halves = {
        "2e mi-temps avec plus de buts": float((g2 > g1).mean()),
        "Autant de buts dans les 2 mi-temps": float((g2 == g1).mean()),
        "1re mi-temps avec plus de buts": float((g1 > g2).mean()),
        "Retournement (mené à la pause, gagne)": float(
            (((ht == 2) & (r == 0)) | ((ht == 0) & (r == 2))).mean()),
        "Match nul à la pause": float((ht == 1).mean()),
    }
    ex_rows = []
    for i in range(min(12, n)):
        ex_rows.append((f"{hg[i]}-{ag[i]}", f"{h1[i]}-{a1[i]}", f"{hg[i]-h1[i]}-{ag[i]-a1[i]}"))
    examples = pd.DataFrame(ex_rows, columns=["Score final", "1re mi-temps", "2e mi-temps"])

    return {"bets": bets, "top": top, "goals": goals, "htft": htft, "halves": halves,
            "examples": examples, "exact": ex,
            "mean_goals": float(tot.mean()), "p_top": float(top["Simulé"].iloc[0]),
            "top_score": top["Score"].iloc[0]}


def predict_row(matrix, n=10000, share=0.44, seed=42):
    """Une ligne de liste de matchs : probabilités 1/N/2 simulées, pronostic, score prévu.

    Le score prévu est le score le plus fréquent PARMI ceux qui correspondent au pronostic 1/N/2
    (le score et le pronostic sont donc toujours cohérents)."""
    sim = simulate(matrix, n=n, share=share, seed=seed)
    hg, ag = sim["hg"], sim["ag"]
    r = _res(hg, ag)
    p = np.bincount(r, minlength=3) / sim["n"]
    pick = int(np.argmax(p))
    sel = r == pick
    vals, cnt = np.unique(hg[sel] * 20 + ag[sel], return_counts=True)
    best = int(vals[np.argmax(cnt)])
    tot = hg + ag
    return {"p": p, "pick": pick, "score": (best // 20, best % 20),
            "p_score": float(cnt.max() / sim["n"]),
            "p_o25": float((tot > 2).mean()), "p_gg": float(((hg > 0) & (ag > 0)).mean())}
