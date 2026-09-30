#!/usr/bin/env python3
"""
backtest_lag.py - raporun tarifine gore yeniden kurulmus backtest (senin kodun DEGIL).
Skor: 1/5/21/63/126/252 gunluk getiri -> gunluk kesit p1-p99 clip -> z-score -> sigmoid -> katsayi ile toplam.
Simulasyon: Top-2 portfoy, rank>10 olan fon 2 gun sonra satilir, yerine en iyi sirali fon alinir.
--lag L: sinyal gununden L gun sonraki fiyattan islem (0 = ayni gun kapanis, 1 = ertesi gun).

Kullanim: python backtest_lag.py price_history_6y.json [--lags 0,1,2]
"""
import argparse, json
import numpy as np
import pandas as pd

LAGS = [1, 5, 21, 63, 126, 252]
COEF = [1, 1, 1, 1, 1, 2.5]
REPORT = [134.19, 22.32, 81.56, 37.86, 29.19]  # raporun baseline test getirileri


def load(path):
    raw = json.load(open(path, encoding="utf-8"))
    idx = pd.to_datetime(raw["tarihler"])
    df = pd.DataFrame({k: pd.Series(v, index=idx) for k, v in raw["fiyatlar"].items()})
    return df.sort_index().replace(0.0, np.nan)


def rank_table(px):
    M = [px / px.shift(k) - 1 for k in LAGS]
    ok = px.notna()
    for m in M:
        ok &= m.notna()
    S = 0
    for c, m in zip(COEF, M):
        m = m.where(ok)
        m = m.clip(lower=m.quantile(0.01, axis=1), upper=m.quantile(0.99, axis=1), axis=0)
        z = m.sub(m.mean(axis=1), axis=0).div(m.std(axis=1), axis=0)
        S = S + c / (1 + np.exp(-z))
    return S.rank(axis=1, ascending=False, method="first")


def simulate(P, R, s, e, L):
    def best(day, excl):
        r = R[day].copy()
        r[list(excl)] = np.inf
        f = int(np.argmin(r))
        return f if np.isfinite(r[f]) else None

    hold = list(np.argsort(R[s])[:2])
    units = [0.5 / P[s + L, f] for f in hold]
    cash, pend = [0.0, 0.0], [None, None]
    for d in range(s + 1, e + 1):
        for i in range(2):
            if hold[i] is None:
                continue
            if pend[i] is None and R[d][hold[i]] > 10:
                pend[i] = d + 2 + L
            if pend[i] == d:
                money = units[i] * P[d, hold[i]]
                other = [h for h in hold if h is not None and h != hold[i]]
                nf = best(max(d - L, s), other)
                if nf is None:
                    cash[i], hold[i], units[i] = money, None, 0.0
                else:
                    hold[i], units[i] = nf, money / P[d, nf]
                pend[i] = None
    val = sum(u * P[e, h] for u, h in zip(units, hold) if h is not None) + sum(cash)
    return (val - 1) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prices")
    ap.add_argument("--lags", default="0,1,2")
    a = ap.parse_args()
    px = load(a.prices)
    R = np.where(np.isnan(rank_table(px).values), np.inf, rank_table(px).values)
    P = px.ffill().values
    n = len(px)
    wins = [(s, s + 126) for s in range(504, n - 126, 126)]
    rows = []
    for k, (s, e) in enumerate(wins):
        row = {"P": k + 1, "test_bas": px.index[s].date(), "test_bit": px.index[e].date(),
               "rapor": REPORT[k] if k < len(REPORT) else np.nan}
        for L in [int(x) for x in a.lags.split(",")]:
            row[f"lag{L}"] = simulate(P, R, s, e, L)
        rows.append(row)
    df = pd.DataFrame(rows)
    print(df.round(1).to_string(index=False))
    print("\nOrtalama:", df.drop(columns=["P", "test_bas", "test_bit"]).mean().round(1).to_dict())


if __name__ == "__main__":
    main()
