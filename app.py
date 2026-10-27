import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from dixon_coles import LEAGUES, load_league, DixonColes
from fixtures import load_fixtures, get_odds

st.set_page_config(page_title="Score Exact Pro", layout="wide")

W_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

VARIANTS = {
    "Buts seuls": dict(xi=0.0018, shots_w=0.0),
    "Mix 50% tirs": dict(xi=0.0018, shots_w=0.5),
    "Tirs seuls": dict(xi=0.0018, shots_w=1.0),
    "Mix 50% + oubli rapide": dict(xi=0.0035, shots_w=0.5),
}


@st.cache_data(ttl=43200, show_spinner="Chargement des données...")
def get_data(code):
    return load_league(code, n_seasons=4)


@st.cache_resource(show_spinner="Calcul du modèle...")
def get_model(code):
    return DixonColes().fit(get_data(code))


@st.cache_data(ttl=3600)
def get_fixtures(code):
    fx = load_fixtures()
    if fx.empty:
        return fx
    fx = fx[fx.Div == code]
    # garde uniquement les matchs pas encore joués
    today = pd.Timestamp.today().normalize()
    fx = fx[fx.Date.isna() | (fx.Date >= today)]
    # exclut les matchs déjà présents dans l'historique (évite la fuite de données)
    df = get_data(code)
    played = set(zip(df.HomeTeam, df.AwayTeam, df.Date.dt.date))
    keep = []
    for h, a, d in zip(fx.HomeTeam, fx.AwayTeam, fx.Date):
        dd = d.date() if pd.notna(d) else None
        keep.append((h, a, dd) not in played)
    return fx[keep].reset_index(drop=True)


def compute_elo(df, k=20, home_adv=60):
    elo = {}
    for h, a, hg, ag in zip(df.HomeTeam, df.AwayTeam, df.FTHG, df.FTAG):
        eh, ea = elo.get(h, 1500), elo.get(a, 1500)
        exp = 1 / (1 + 10 ** (-(eh + home_adv - ea) / 400))
        res = 1 if hg > ag else 0.5 if hg == ag else 0
        gd = abs(hg - ag)
        mult = 1 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8
        d = k * mult * (res - exp)
        elo[h], elo[a] = eh + d, ea - d
    return elo


def team_form(df, team, n=5):
    m = df[(df.HomeTeam == team) | (df.AwayTeam == team)].tail(n)
    rows = []
    for _, r in m.iterrows():
        home = r.HomeTeam == team
        gf, ga = (r.FTHG, r.FTAG) if home else (r.FTAG, r.FTHG)
        res = "V" if gf > ga else "N" if gf == ga else "D"
        rows.append({"Date": r.Date.date(), "Lieu": "Dom" if home else "Ext",
                     "Adversaire": r.AwayTeam if home else r.HomeTeam,
                     "Score": f"{int(gf)}-{int(ga)}", "Résultat": res})
    return pd.DataFrame(rows)


def h2h(df, home, away, n=6):
    m = df[((df.HomeTeam == home) & (df.AwayTeam == away)) |
           ((df.HomeTeam == away) & (df.AwayTeam == home))].tail(n)
    return m[["Date", "HomeTeam", "FTHG", "FTAG", "AwayTeam"]].assign(
        Date=lambda d: d.Date.dt.date)


def probs_1x2(M):
    return np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()


def fmt_date(d):
    return d.strftime("%d/%m") if pd.notna(d) else "--/--"


def blend_probs(p_model, odds, w):
    """Mélange modèle / marché. Sans cotes : modèle seul."""
    if odds is None:
        return p_model, None
    q = 1 / np.array(odds, dtype=float)
    q = q / q.sum()
    return w * p_model + (1 - w) * q, q


def adjust_matrix(M, p_target):
    """Recale la matrice des scores pour que le 1X2 colle au mélange."""
    p_model = np.array(probs_1x2(M))
    f = p_target / np.clip(p_model, 1e-12, None)
    idx = np.indices(M.shape)
    i, j = idx[0], idx[1]
    F = np.where(i > j, f[0], np.where(i == j, f[1], f[2]))
    A = M * F
    return A / A.sum()


def analyse(model, home, away, odds, w, top=5):
    M = model.score_matrix(home, away)
    lh, la = model.rates(home, away)
    p_model = np.array(probs_1x2(M))
    p_blend, q = blend_probs(p_model, odds, w)
    A = adjust_matrix(M, p_blend)
    n = A.shape[0]
    flat = sorted(((i, j, A[i, j]) for i in range(n) for j in range(n)),
                  key=lambda x: -x[2])[:top]
    over25 = sum(A[i, j] for i in range(n) for j in range(n) if i + j > 2)
    btts = 1 - A[0, :].sum() - A[:, 0].sum() + A[0, 0]
    return {"A": A, "lh": lh, "la": la, "p_model": p_model, "q": q,
            "p_blend": p_blend,
            "top": [(f"{i}-{j}", p * 100) for i, j, p in flat],
            "over25": over25 * 100, "btts": btts * 100}


@st.cache_data(show_spinner="Backtest en cours (3 à 6 min)...")
def backtest(code, n_test=90, step=30):
    df = get_data(code)
    start = len(df) - n_test
    recs = []
    for s in range(0, n_test, step):
        chunk = df.iloc[start + s: start + s + step]
        ref = chunk.Date.min()
        train = df.iloc[:start + s]
        models = {name: DixonColes(**kw).fit(train, ref_date=ref)
                  for name, kw in VARIANTS.items()}
        first = next(iter(models.values()))
        for _, r in chunk.iterrows():
            if r.HomeTeam not in first.idx or r.AwayTeam not in first.idx:
                continue
            odds = np.array([r.get("AvgH", np.nan), r.get("AvgD", np.nan),
                             r.get("AvgA", np.nan)], dtype=float)
            if np.isnan(odds).any():
                continue
            q = 1 / odds
            q = q / q.sum()
            actual = 0 if r.FTHG > r.FTAG else 1 if r.FTHG == r.FTAG else 2
            p = {name: np.array(probs_1x2(m.score_matrix(r.HomeTeam, r.AwayTeam)))
                 for name, m in models.items()}
            recs.append({"q": q, "actual": actual, "p": p})
    if not recs:
        return None
    onehot = np.eye(3)
    br_market = np.array([((x["q"] - onehot[x["actual"]]) ** 2).sum() for x in recs])
    rows = []
    for name in VARIANTS:
        br_model = None
        best = None
        for w in W_GRID:
            bw = np.array([((w * x["p"][name] + (1 - w) * x["q"]
                             - onehot[x["actual"]]) ** 2).sum() for x in recs])
            mean = float(bw.mean())
            if w == 1.0:
                br_model = mean
            if best is None or mean < best["brier"]:
                d = bw - br_market
                se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
                best = {"w": w, "brier": mean, "diff": float(d.mean()), "se": se}
        rows.append({"name": name, "brier_model": br_model, **best})
    return {"n": len(recs), "brier_market": float(br_market.mean()), "rows": rows}


def scan_fixtures(model, fx, w):
    rows = []
    for _, f in fx.iterrows():
        h, a = f.HomeTeam, f.AwayTeam
        if h not in model.idx or a not in model.idx:
            continue
        od = get_odds(f)
        res = analyse(model, h, a, od, w, top=3)
        pb = res["p_blend"] * 100
        row = {"Date": fmt_date(f.Date), "Match": f"{h} - {a}",
               "Buts att.": f"{res['lh']:.2f}-{res['la']:.2f}",
               "Score 1": f"{res['top'][0][0]} ({res['top'][0][1]:.1f}%)",
               "Score 2": f"{res['top'][1][0]} ({res['top'][1][1]:.1f}%)",
               "Score 3": f"{res['top'][2][0]} ({res['top'][2][1]:.1f}%)",
               "1/X/2 %": f"{pb[0]:.0f}/{pb[1]:.0f}/{pb[2]:.0f}",
               "Over 2.5 %": round(res["over25"], 1),
               "BTTS %": round(res["btts"], 1),
               "Meilleur EV": np.nan, "Pari": "-", "Value": ""}
        if od:
            ev = res["p_blend"] * np.array(od)
            k = int(ev.argmax())
            row["Meilleur EV"] = round(float(ev[k]), 2)
            row["Pari"] = ["1", "X", "2"][k]
            row["Value"] = "OUI" if ev[k] > 1.05 else ""
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------- Interface ----------------
st.title("⚽ Score Exact Pro")

code = st.sidebar.selectbox("Championnat", list(LEAGUES),
                            format_func=lambda c: f"{LEAGUES[c]} ({c})",
                            index=list(LEAGUES).index("F1"))
df = get_data(code)
model = get_model(code)
fx = get_fixtures(code)
recent = sorted(set(df[df.Date > df.Date.max() - pd.Timedelta(days=400)].HomeTeam))

mode = "Manuel"
if not fx.empty:
    mode = st.sidebar.radio("Source du match", ["Matchs à venir", "Manuel"])

odds_known = False
if mode == "Matchs à venir":
    labels = [f"{fmt_date(r.Date)}  {r.HomeTeam} - {r.AwayTeam}"
              for r in fx.itertuples()]
    i = st.sidebar.selectbox("Match", range(len(fx)), format_func=lambda k: labels[k])
    row = fx.iloc[i]
    home, away = row.HomeTeam, row.AwayTeam
    od = get_odds(row)
    if od is None:
        st.sidebar.caption("Cotes indisponibles pour ce match : saisis-les à la main.")
        od = (2.00, 3.40, 3.60)
    else:
        odds_known = True
else:
    home = st.sidebar.selectbox("Domicile", recent, index=0)
    away = st.sidebar.selectbox("Extérieur", recent, index=min(1, len(recent) - 1))
    od = (2.00, 3.40, 3.60)

st.sidebar.markdown("**Cotes du bookmaker**")
kk = f"{home}_{away}_{mode}"
o1 = st.sidebar.number_input("Cote 1", 1.01, 100.0, float(od[0]), 0.05, key=f"o1_{kk}")
oX = st.sidebar.number_input("Cote X", 1.01, 100.0, float(od[1]), 0.05, key=f"oX_{kk}")
o2 = st.sidebar.number_input("Cote 2", 1.01, 100.0, float(od[2]), 0.05, key=f"o2_{kk}")

use_odds = odds_known
if not odds_known:
    use_odds = st.sidebar.checkbox("J'ai saisi les vraies cotes ci-dessus",
                                   value=False)

w = st.sidebar.slider("Poids du modèle (0 = marché seul, 1 = modèle seul)",
                      0.0, 1.0, 0.3, 0.1)

tab0, tab1, tab2, tab3 = st.tabs(
    ["Matchs à venir", "Prédiction", "Forme & Elo", "Backtest"])

with tab0:
    if fx.empty:
        st.info("Aucun match à venir pour cette ligue : le fichier ne contient "
                "que des matchs déjà joués, ou n'est pas encore mis à jour. "
                "Utilise le mode Manuel dans la barre latérale.")
    else:
        scan = scan_fixtures(model, fx, w)
        if scan.empty:
            st.warning("Équipes absentes de l'historique (promus ?).")
        else:
            st.dataframe(scan.sort_values("Meilleur EV", ascending=False),
                         use_container_width=True, hide_index=True)
            st.caption("Probabilités = mélange modèle/marché selon le poids "
                       "choisi. Value = EV > 1.05 : valide d'abord avec le backtest.")

with tab1:
    if home == away:
        st.warning("Choisis deux équipes différentes.")
    elif home not in model.idx or away not in model.idx:
        st.warning("Une des équipes n'est pas dans l'historique (promu récent ?).")
    else:
        odds_arg = (o1, oX, o2) if use_odds else None
        res = analyse(model, home, away, odds_arg, w)
        c1, c2, c3 = st.columns(3)
        c1.metric("Buts attendus (modèle)", f"{res['lh']:.2f} - {res['la']:.2f}")
        c2.metric("Over 2.5", f"{res['over25']:.1f}%")
        c3.metric("Les deux marquent", f"{res['btts']:.1f}%")

        if use_odds:
            st.caption(f"Scores recalés sur un mélange {w:.0%} modèle / "
                       f"{1 - w:.0%} marché.")
        else:
            st.caption("Modèle seul : entre les vraies cotes dans la barre "
                       "latérale pour activer le mélange avec le marché.")

        st.subheader(f"{home} vs {away} : top 5 des scores exacts")
        top_df = pd.DataFrame(res["top"], columns=["Score", "Probabilité %"])
        st.table(top_df.style.format({"Probabilité %": "{:.1f}"}))

        g = res["A"][:7, :7] * 100
        fig = px.imshow(g, text_auto=".1f", color_continuous_scale="Blues",
                        labels=dict(x=f"Buts {away}", y=f"Buts {home}", color="%"))
        fig.update_xaxes(fixedrange=True)
        fig.update_yaxes(fixedrange=True)
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})

        st.subheader("Value bet")
        if use_odds:
            odds_np = np.array([o1, oX, o2])
            marge = (1 / odds_np).sum() - 1
            tab = pd.DataFrame({"Issue": ["1", "X", "2"],
                                "Modèle %": res["p_model"] * 100,
                                "Marché %": res["q"] * 100,
                                "Mélange %": res["p_blend"] * 100,
                                "EV": res["p_blend"] * odds_np})
            st.table(tab.style.format({"Modèle %": "{:.1f}", "Marché %": "{:.1f}",
                                       "Mélange %": "{:.1f}", "EV": "{:.2f}"}))
            st.caption(f"Marge du bookmaker : {marge * 100:.1f}%. EV calculée avec "
                       "les probabilités du mélange. EV > 1.05 = valeur potentielle, "
                       "à confirmer avec le backtest.")
        else:
            st.info("Coche « J'ai saisi les vraies cotes » dans la barre latérale "
                    "pour afficher la comparaison.")

with tab2:
    elo = compute_elo(df)
    st.subheader("Classement Elo")
    e = pd.DataFrame({"Équipe": recent, "Elo": [round(elo.get(t, 1500)) for t in recent]})
    st.dataframe(e.sort_values("Elo", ascending=False).reset_index(drop=True),
                 use_container_width=True)
    ca, cb = st.columns(2)
    ca.subheader(f"Forme {home}")
    ca.dataframe(team_form(df, home), use_container_width=True)
    cb.subheader(f"Forme {away}")
    cb.dataframe(team_form(df, away), use_container_width=True)
    st.subheader("Confrontations directes")
    st.dataframe(h2h(df, home, away), use_container_width=True)

with tab3:
    st.write("Compare plusieurs versions du modèle (buts seuls, tirs, oubli rapide) "
             "au marché, avec et sans mélange. Chaque prédiction n'utilise que les "
             "matchs précédents.")
    cov = (df[["HS", "HST"]].notna().all(axis=1).mean()
           if all(c in df.columns for c in ("HS", "HST")) else 0.0)
    st.caption(f"Matchs avec données de tirs : {cov:.0%}")
    if cov < 0.5:
        st.warning("Peu ou pas de données de tirs pour cette ligue : les variantes "
                   "« tirs » seront presque identiques à « Buts seuls ».")
    n_test = st.selectbox("Nombre de matchs testés", [90, 180, 300], index=0)
    if st.button("Lancer le backtest"):
        b = backtest(code, n_test=n_test)
        if not b:
            st.error("Pas assez de données avec cotes.")
        else:
            c1, c2 = st.columns(2)
            c1.metric("Matchs testés", b["n"])
            c2.metric("Brier du marché", f"{b['brier_market']:.4f}")

            t = pd.DataFrame([{
                "Variante": r["name"],
                "Modèle seul": r["brier_model"],
                "Meilleur poids": r["w"],
                "Mélange": r["brier"],
                "Écart vs marché": f"{r['diff']:+.4f} ± {r['se']:.4f}",
            } for r in b["rows"]])
            st.table(t.style.format({"Modèle seul": "{:.4f}", "Mélange": "{:.4f}",
                                     "Meilleur poids": "{:.1f}"}))

            wins = [r for r in b["rows"] if r["w"] > 0 and r["diff"] < -2 * r["se"]]
            if wins:
                names = ", ".join(r["name"] for r in wins)
                st.success(f"Gain significatif face au marché : {names}. À vérifier "
                           "sur d'autres ligues avant de t'y fier.")
            else:
                st.warning("Aucune variante ne bat le marché de façon significative "
                           "sur cet échantillon.")
            st.caption("Brier : plus bas = mieux. « Écart vs marché » est négatif "
                       "quand le mélange fait mieux que le marché ; il compte "
                       "seulement s'il dépasse 2 fois l'erreur (±). Le meilleur "
                       "poids est choisi sur ces mêmes matchs, donc l'écart reste "
                       "optimiste.")
