#!/usr/bin/env python3
"""
benchmark_kontrol.py - walk-forward sonuclarini sert null'larla karsilastirir ve veriyi kontrol eder.

Ornek:
  python benchmark_kontrol.py price_history_6y.json \
      --baseline 134.19,22.32,81.56,37.86,29.19 --cash-fund PPF1 --old price_history_temiz.json

Yaptiklari:
  1) Veri ozeti: ilk/son tarih, fon sayisi, kapsama (son tarih bayat mi?)
  2) Olcek sicramalari: gunluk oran <0.5 veya >2 olan yerler, 10^k'ya yakinlar isaretli
  3) (--old) Yeni ve eski dosyada ortak gunlerde fiyat farklari
  4) Pencere sayisi ve tarihleri (warmup + train + test aritmetigi)
  5) Her test penceresi icin: evren esit agirlikli ortalama, evren medyani, rastgele 2 fon
     (al-tut) dagilimi, opsiyonel nakit fonu, ve --baseline getirisinin bu dagilimdaki yuzdeligi

Not: rastgele-2-fon null'i al-tut'tur; sizin simulasyonunuz (rank>10 ise 2 gun sonra sat)
yeniden dengeleme yapiyor. Birebir kiyas icin ayni simulasyonu rastgele skorlarla kosmak daha
dogru olur, bu script ilk, hizli bir es-gecme testidir.
"""
import argparse, json
import numpy as np
import pandas as pd

DATE_KEYS = ("tarih", "date", "Tarih", "Date")
PRICE_KEYS = ("fiyat", "price", "Fiyat", "Price", "deger", "value")
CODE_KEYS = ("kod", "code", "fon", "fund", "Kod", "Fon")


def pick(d, keys):
    for k in keys:
        if k in d:
            return d[k]
    raise KeyError(f"beklenen anahtar yok {keys}; eldeki: {list(d)[:6]}")


def to_series(v):
    if isinstance(v, dict):
        pairs = list(v.items())
    else:
        pairs = [(pick(e, DATE_KEYS), pick(e, PRICE_KEYS)) if isinstance(e, dict) else (e[0], e[1]) for e in v]
    s = pd.Series({pd.to_datetime(d, dayfirst="." in str(d)): float(p) for d, p in pairs}).sort_index()
    return s[~s.index.duplicated()]


def load(path):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        data = {k: to_series(v) for k, v in raw.items()}
    else:  # kayit listesi: [{kod, tarih, fiyat}, ...]
        g = {}
        for r in raw:
            g.setdefault(pick(r, CODE_KEYS), []).append(r)
        data = {k: to_series(v) for k, v in g.items()}
    return pd.DataFrame(data).sort_index()


def scale_report(px):
    print("\n== Olcek sicramalari (gunluk oran <0.5 veya >2) ==")
    found = 0
    for c in px.columns:
        s = px[c].dropna()
        lg = np.log10(s / s.shift())
        for d in lg.index[lg.abs() > 0.3]:
            l = lg[d]
            k = round(l)
            tag = f"  ~10^{k}" if k != 0 and abs(l - k) < 0.05 else ""
            i = s.index.get_loc(d)
            print(f"{c:8s} {d.date()} {s.iloc[i-1]:.6g} -> {s.iloc[i]:.6g} (oran {10**l:.4g}){tag}")
            found += 1
    if not found:
        print("yok (temiz)")


def overlap_report(new, old_path, tol=0.005):
    old = load(old_path)
    cols, idx = new.columns.intersection(old.columns), new.index.intersection(old.index)
    a, b = new.loc[idx, cols], old.loc[idx, cols]
    rel = (a / b - 1).abs().where(a.notna() & b.notna())
    print(f"\n== Eski dosyayla ortak gunler: {len(cols)} fon, {len(idx)} gun, tolerans %{tol*100:.1f} ==")
    cnt = (rel > tol).sum()
    cnt = cnt[cnt > 0].sort_values(ascending=False)
    if cnt.empty:
        print("fark yok")
        return
    for c in cnt.index[:15]:
        ratio = (a[c] / b[c]).median()
        note = "  (sabit carpan: olcek duzeltmesi olabilir)" if (a[c] / b[c]).std() < 1e-6 and abs(ratio - 1) > tol else ""
        print(f"{c:8s} {cnt[c]} gun farkli, en buyuk %{rel[c].max()*100:.2f}, medyan oran {ratio:.4g}{note}")


def make_windows(n, warm, train, test, anchor):
    first = warm + train
    if anchor == "start":
        return [(s, s + test) for s in range(first, n - test, test)]
    w, e = [], n - 1
    while e - test >= first:
        w.append((e - test, e))
        e -= test
    return w[::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prices")
    ap.add_argument("--train", type=int, default=252)
    ap.add_argument("--test", type=int, default=126)
    ap.add_argument("--warmup", type=int, default=252, help="252 gunluk getiri metrigi icin isinma")
    ap.add_argument("--anchor", choices=["start", "end"], default="end")
    ap.add_argument("--draws", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baseline", help="pencere basina baseline getirileri (%%), virgulle")
    ap.add_argument("--cash-fund", help="nakit proxy'si olarak kullanilacak fon kodu (para piyasasi)")
    ap.add_argument("--old", help="eski fiyat dosyasi (ortak gun kontrolu)")
    ap.add_argument("--out", default="benchmark_penceler.csv")
    a = ap.parse_args()

    px = load(a.prices)
    n = len(px)
    print(f"Veri: {px.index[0].date()} -> {px.index[-1].date()} | {px.shape[1]} fon | {n} gun")
    if (pd.Timestamp.today().normalize() - px.index[-1]).days > 7:
        print(f"UYARI: son tarih bugunden {(pd.Timestamp.today().normalize() - px.index[-1]).days} gun geride")
    full = (px.apply(pd.Series.first_valid_index) <= px.index[0] + pd.Timedelta(days=10)).sum()
    print(f"Tam gecmisi olan fon: {full}/{px.shape[1]}")

    scale_report(px)
    if a.old:
        overlap_report(px, a.old)

    wins = make_windows(n, a.warmup, a.train, a.test, a.anchor)
    print(f"\n== {len(wins)} pencere (warmup {a.warmup}, train {a.train}, test {a.test}, anchor={a.anchor}) ==")
    if not wins:
        return
    bl = {}
    if a.baseline:
        b = [float(x) for x in a.baseline.split(",")]
        ids = list(range(len(wins)))
        ids = ids[-len(b):] if a.anchor == "end" else ids[:len(b)]
        if len(b) != len(wins):
            print(f"UYARI: {len(b)} baseline degeri var, {len(wins)} pencere var; hizalama anchor={a.anchor}")
        bl = dict(zip(ids, b))

    rng = np.random.default_rng(a.seed)
    rows = []
    for k, (s, e) in enumerate(wins):
        ok = px.iloc[s - a.train].notna() & px.iloc[s].notna() & px.iloc[e].notna()
        ret = ((px.iloc[e] / px.iloc[s] - 1) * 100)[ok]
        m = len(ret)
        idx = np.argpartition(rng.random((a.draws, m)), 2, axis=1)[:, :2]
        r2 = ret.values[idx].mean(axis=1)
        row = {"P": k + 1, "test_bas": px.index[s].date(), "test_bit": px.index[e].date(), "fon": m,
               "evren_ort": ret.mean(), "evren_med": ret.median(),
               "rnd2_p10": np.percentile(r2, 10), "rnd2_med": np.percentile(r2, 50), "rnd2_p90": np.percentile(r2, 90)}
        if a.cash_fund:
            row["nakit"] = (px[a.cash_fund].iloc[e] / px[a.cash_fund].iloc[s] - 1) * 100
        if k in bl:
            row["baseline"] = bl[k]
            row["bl_yuzdelik_rnd2"] = (r2 < bl[k]).mean() * 100
        rows.append(row)

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.round(1).to_string(index=False))
    df.to_csv(a.out, index=False)

    if bl:
        d = df.dropna(subset=["baseline"])
        print(f"\nBaseline > evren ort: {(d.baseline > d.evren_ort).sum()}/{len(d)} | > rnd2 medyan: {(d.baseline > d.rnd2_med).sum()}/{len(d)}"
              + (f" | > nakit: {(d.baseline > d.nakit).sum()}/{len(d)}" if a.cash_fund else ""))
    print(f"\nKaydedildi: {a.out}")


if __name__ == "__main__":
    main()
