import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from dixon_coles import LEAGUES, load_league, DixonColes
from fixtures import load_fixtures, get_odds

st.set_page_config(page_title="Score Exact Pro", layout="wide")


@st.cache_data(ttl=43200, show_spinner="Chargement des données...")
def get_data(code):
    return load_league(code)


@st.cache_resource(show_spinner="Calcul du modèle...")
def get_model(code):
    return DixonColes().fit(get_data(code))


@st.cache_data(ttl=3600)
def get_fixtures(code):
    fx = load_fixtures()
    if fx.empty:
        return fx
    return fx[fx.Div == code].reset_index(drop=True)


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


@st.cache_data(show_spinner="Backtest en cours (1 à 2 min)...")
def backtest(code, n_test=100, step=20):
    df = get_data(code)
    start = len(df) - n_test
    rows = []
    for s in range(0, n_test, step):
        chunk = df.iloc[start + s: start + s + step]
        m = DixonColes().fit(df.iloc[:start + s], ref_date=chunk.Date.min())
        for _, r in chunk.iterrows():
            if r.HomeTeam not in m.idx or r.AwayTeam not in m.idx:
                continue
            M = m.score_matrix(r.HomeTeam, r.AwayTeam)
            p = np.array(probs_1x2(M))
            actual = 0 if r.FTHG > r.FTAG else 1 if r.FTHG == r.FTAG else 2
            i, j = np.unravel_index(M.argmax(), M.shape)
            odds = np.array([r.get("AvgH", np.nan), r.get("AvgD", np.nan),
                             r.get("AvgA", np.nan)], dtype=float)
            rows.append({"p": p, "actual": actual, "odds": odds,
                         "exact_hit": (i == r.FTHG and j == r.FTAG)})
    if not rows:
        return None
    onehot = lambda a: np.eye(3)[a]
    brier = np.mean([((x["p"] - onehot(x["actual"])) ** 2).sum() for x in rows])
    acc = np.mean([x["p"].argmax() == x["actual"] for x in rows])
    exact = np.mean([x["exact_hit"] for x in rows])
    out = {"n": len(rows), "brier": brier, "accuracy": acc, "exact": exact,
           "brier_book": None, "roi": None, "bets": 0}
    ok = [x for x in rows if not np.isnan(x["odds"]).any()]
    if ok:
        bb = []
        for x in ok:
            q = 1 / x["odds"]
            q = q / q.sum()
            bb.append(((q - onehot(x["actual"])) ** 2).sum())
        out["brier_book"] = np.mean(bb)
        profit, bets = 0.0, 0
        for x in ok:
            ev = x["p"] * x["odds"]
            k = ev.argmax()
            if ev[k] > 1.05:
                bets += 1
                profit += (x["odds"][k] - 1) if x["actual"] == k else -1
        out["bets"] = bets
        out["roi"] = profit / bets * 100 if bets else None
    return out


def scan_fixtures(model, fx):
    rows = []
    for _, f in fx.iterrows():
        h, a = f.HomeTeam, f.AwayTeam
        if h not in model.idx or a not in model.idx:
            continue
        r = model.summary(h, a, top=3)
        od = get_odds(f)
        p = np.array([r["1"], r["X"], r["2"]]) / 100
        row = {"Date": fmt_date(f.Date), "Match": f"{h} - {a}",
               "Buts att.": f"{r['buts_attendus'][0]}-{r['buts_attendus'][1]}",
               "Score 1": f"{r['top_scores'][0][0]} ({r['top_scores'][0][1]}%)",
               "Score 2": f"{r['top_scores'][1][0]} ({r['top_scores'][1][1]}%)",
               "Score 3": f"{r['top_scores'][2][0]} ({r['top_scores'][2][1]}%)",
               "1/X/2 %": f"{r['1']:.0f}/{r['X']:.0f}/{r['2']:.0f}",
               "Over 2.5 %": r["over_2.5"], "BTTS %": r["btts"],
               "Meilleur EV": np.nan, "Pari": "-", "Value": ""}
        if od:
            ev = p * np.array(od)
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
    home = st.sidebar.selectbox("Domicile", recent, index=0)
    away = st.sidebar.selectbox("Extérieur", recent, index=min(1, len(recent) - 1))
    od = (2.00, 3.40, 3.60)

st.sidebar.markdown("**Cotes du bookmaker**")
kk = f"{home}_{away}_{mode}"
o1 = st.sidebar.number_input("Cote 1", 1.01, 100.0, float(od[0]), 0.05, key=f"o1_{kk}")
oX = st.sidebar.number_input("Cote X", 1.01, 100.0, float(od[1]), 0.05, key=f"oX_{kk}")
o2 = st.sidebar.number_input("Cote 2", 1.01, 100.0, float(od[2]), 0.05, key=f"o2_{kk}")

tab0, tab1, tab2, tab3 = st.tabs(
    ["Matchs à venir", "Prédiction", "Forme & Elo", "Backtest"])

with tab0:
    if fx.empty:
        st.info("Aucun match à venir trouvé pour cette ligue (fixtures.csv ne "
                "couvre que la semaine en cours, et peut être vide entre deux "
                "journées). Utilise le mode Manuel.")
    else:
        scan = scan_fixtures(model, fx)
        if scan.empty:
            st.warning("Équipes absentes de l'historique (promus ?).")
        else:
            st.dataframe(scan.sort_values("Meilleur EV", ascending=False),
                         use_container_width=True, hide_index=True)
            st.caption("Value = EV > 1.05 sur le 1X2. Un EV élevé peut aussi "
                       "venir d'un modèle qui se trompe : valide avec le backtest.")

with tab1:
    if home == away:
        st.warning("Choisis deux équipes différentes.")
    elif home not in model.idx or away not in model.idx:
        st.warning("Une des équipes n'est pas dans l'historique (promu récent ?).")
    else:
        M = model.score_matrix(home, away)
        r = model.summary(home, away, top=5)
        c1, c2, c3 = st.columns(3)
        c1.metric("Buts attendus", f"{r['buts_attendus'][0]} - {r['buts_attendus'][1]}")
        c2.metric("Over 2.5", f"{r['over_2.5']}%")
        c3.metric("Les deux marquent", f"{r['btts']}%")

        st.subheader(f"{home} vs {away} : top 5 des scores exacts")
        st.table(pd.DataFrame(r["top_scores"], columns=["Score", "Probabilité %"]))

        g = M[:7, :7] * 100
        fig = px.imshow(g, text_auto=".1f", color_continuous_scale="Blues",
                        labels=dict(x=f"Buts {away}", y=f"Buts {home}", color="%"))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Value bet (modèle vs cotes)")
        p = np.array([r["1"], r["X"], r["2"]]) / 100
        q = 1 / np.array([o1, oX, o2])
        marge = q.sum() - 1
        q = q / q.sum()
        tab = pd.DataFrame({"Issue": ["1", "X", "2"],
                            "Modèle %": (p * 100).round(1),
                            "Bookmaker %": (q * 100).round(1),
                            "Écart": ((p - q) * 100).round(1),
                            "EV": (p * np.array([o1, oX, o2])).round(2)})
        st.table(tab)
        st.caption(f"Marge du bookmaker : {marge * 100:.1f}%. EV > 1.05 = valeur "
                   "potentielle, à confirmer avec le backtest.")

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
    st.write("Teste le modèle sur les 100 derniers matchs, chaque prédiction "
             "n'utilisant que les matchs précédents.")
    if st.button("Lancer le backtest"):
        b = backtest(code)
        if not b:
            st.error("Pas assez de données.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Matchs testés", b["n"])
            c2.metric("Précision 1X2", f"{b['accuracy'] * 100:.1f}%")
            c3.metric("Score exact trouvé", f"{b['exact'] * 100:.1f}%")
            c4, c5 = st.columns(2)
            c4.metric("Brier modèle", f"{b['brier']:.3f}")
            if b["brier_book"]:
                c5.metric("Brier bookmakers", f"{b['brier_book']:.3f}")
                verdict = ("Le modèle bat le marché" if b["brier"] < b["brier_book"]
                           else "Les bookmakers restent meilleurs (plus bas = mieux)")
                st.info(verdict)
            if b["roi"] is not None:
                st.metric(f"ROI simulé ({b['bets']} paris, EV > 1.05)", f"{b['roi']:.1f}%")
            st.caption("100 matchs, c'est peu : un ROI positif ici peut être de la "
                       "chance. Relance sur plusieurs ligues.")
