#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEFAS gerçekçi DE walk-forward backtest (valör + T+1 mekaniği, 9 serbest parametre)

Serbest parametreler (DE):
  6 dönemsel katsayı (gün/hafta/ay/3ay/6ay/yıl), Sortino ağırlığı,
  portföy boyutu P in [2,4] (tamsayı), satış eşiği top_x in [2,12] (tamsayı)
Sabit: Sortino periyodu (--sortino-gun, varsayılan 69), sigmoid k (varsayılan 3)

Mekanik (gün d = karar günü, getiri(d) = fiyat[d]/fiyat[d-1]-1):
  - Elde tutulan fon rank > top_x ise: satış kararı d, satış d+1'de (d+1 getirisini alır)
  - Satış gelirine, satış gününden (valör+1) gün sonra ulaşılır; o gün alış kararı verilir
  - Alış kararının ertesi günü (T+1) fon portföye girer (d+1 fiyatından alır, d+2'den itibaren getiri alır)
  - Nakit beklerken getiri 0. Komisyon yok. Kesirli adet serbest (tam sayı adet modellenmedi).

Kullanım:
  python de_walkforward_standalone.py --inspect
  python de_walkforward_standalone.py --train 252 --test 252
  python de_walkforward_standalone.py --synthetic --popsize 6 --maxiter 8 --reopt-every 10   # duman testi
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import rankdata

LOOKBACKS = [1, 5, 21, 63, 126, 252]
COEF_NAMES = ["gun", "hafta", "ay", "3ay", "6ay", "yil"]
BASE_COEFS = [0.43, 0.67, 2.41, 2.25, 1.55, 2.53]
NCOMP = 7  # 6 dönemsel + sortino
_G = {}     # DE işçi süreçlerinin (fork) paylaştığı veri


# ----------------------------------------------------------------- veri yükleme
def _pick(keys, cands):
    low = {k.lower(): k for k in keys}
    for c in cands:
        if c in low:
            return low[c]
    return None


def _records_to_df(recs):
    keys = list(recs[0].keys())
    kc = _pick(keys, ["code", "fon_kodu", "kod", "fund_code", "fund", "fon"])
    kd = _pick(keys, ["date", "tarih"])
    kp = _pick(keys, ["price", "fiyat", "close", "deger", "value"])
    if not (kc and kd and kp):
        raise ValueError(f"Kayıt anahtarları tanınmadı: {keys}")
    df = pd.DataFrame(recs)[[kc, kd, kp]]
    df.columns = ["code", "date", "price"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.pivot_table(index="date", columns="code", values="price", aggfunc="last")


def _fail(obj):
    print("[HATA] price_history.json yapisi taninmadi. Asagidaki satirlari gonderin:", flush=True)
    print("  ust seviye tip:", type(obj).__name__, "uzunluk:", len(obj) if hasattr(obj, "__len__") else "-", flush=True)
    if isinstance(obj, dict):
        ks = list(obj.keys())[:5]
        print("  ilk anahtarlar:", ks, flush=True)
        v = obj[ks[0]] if ks else None
    elif isinstance(obj, list) and obj:
        v = obj[0]
    else:
        v = None
    if isinstance(obj, dict) and "fiyatlar" in obj:
        fy = obj["fiyatlar"]
        print("  fiyatlar tipi:", type(fy).__name__, flush=True)
        if isinstance(fy, dict):
            k0 = next(iter(fy), None)
            print("  fiyatlar ilk anahtar:", k0, "->", str(fy[k0])[:200], flush=True)
        elif isinstance(fy, list) and fy:
            print("  fiyatlar ilk oge:", str(fy[0])[:200], flush=True)
    print("  ilk oge tipi:", type(v).__name__, flush=True)
    print("  ilk oge ornek:", str(v)[:500], flush=True)
    if isinstance(v, dict):
        print("  ilk oge anahtarlari:", list(v.keys())[:10], flush=True)
    raise SystemExit(1)


def load_prices(path):
    with open(path, encoding="utf-8") as f:
        _obj = json.load(f)
    try:
        return _load_prices(_obj)
    except SystemExit:
        raise
    except Exception as e:
        print("[HATA] ayristirma hatasi:", repr(e)[:300], flush=True)
        _fail(_obj)


def _load_prices(obj):
    if isinstance(obj, dict) and "tarihler" in obj and "fiyatlar" in obj:
        dates = pd.to_datetime(pd.Series(obj["tarihler"]), errors="coerce")
        fy = obj["fiyatlar"]
        if not isinstance(fy, dict):
            raise ValueError("fiyatlar dict degil: " + type(fy).__name__)
        cols = {}
        for c, v in fy.items():
            if isinstance(v, dict):
                cols[c] = pd.Series({pd.to_datetime(k): x for k, x in v.items()})
            else:
                v = list(v)[:len(dates)]
                cols[c] = pd.Series(v + [None] * (len(dates) - len(v)), index=dates.values)
        df = pd.DataFrame(cols)
        df = df[~df.index.isna()]
        df = df[~df.index.duplicated(keep="first")].sort_index()
        df = df.apply(pd.to_numeric, errors="coerce").ffill()
        return df.loc[:, df.notna().sum() > 30]
    for _ in range(3):  # {"data": {...}} gibi tek anahtarlı sarmalayıcıları aç
        if isinstance(obj, dict) and len(obj) == 1:
            obj = next(iter(obj.values()))
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        df = _records_to_df(obj)
    elif isinstance(obj, dict):
        first = next(iter(obj.values()))
        if isinstance(first, dict):
            df = pd.DataFrame(obj)
            df.index = pd.to_datetime(df.index, errors="coerce")
        elif isinstance(first, list) and first and isinstance(first[0], (list, tuple)):
            df = pd.DataFrame({c: pd.Series({pd.to_datetime(a): b for a, b in v}) for c, v in obj.items()})
        elif isinstance(first, list) and first and isinstance(first[0], dict):
            parts = {}
            for c, v in obj.items():
                kk = list(v[0].keys())
                kd, kp = _pick(kk, ["date", "tarih"]), _pick(kk, ["price", "fiyat", "close", "deger", "value"])
                parts[c] = pd.Series({pd.to_datetime(r[kd]): r[kp] for r in v})
            df = pd.DataFrame(parts)
        else:
            _fail(obj)
    else:
        _fail(obj)
    df = df[~df.index.isna()].sort_index()
    df = df.apply(pd.to_numeric, errors="coerce").ffill()
    df = df.loc[:, df.notna().sum() > 30]
    return df


def load_valor(path, codes):
    val = {}
    try:
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        items = []
        if isinstance(obj, dict) and all(isinstance(v, dict) for v in obj.values()):
            items = [dict(v, _code=k) for k, v in obj.items()]
        else:
            while isinstance(obj, dict) and len(obj) == 1:
                obj = next(iter(obj.values()))
            items = obj if isinstance(obj, list) else []
        for it in items:
            keys = list(it.keys())
            kc = _pick(keys, ["_code", "code", "fon_kodu", "kod", "fund_code", "fon"])
            kv = next((k for k in keys if "val" in k.lower()), None)
            if kc and kv and it[kv] is not None:
                val[str(it[kc])] = int(it[kv])
    except Exception as e:
        print(f"[uyarı] valör dosyası okunamadı ({e}); hepsi 2 varsayıldı", flush=True)
    med = int(np.median(list(val.values()))) if val else 2
    missing = [c for c in codes if c not in val]
    if missing:
        print(f"[bilgi] valörü olmayan {len(missing)} fon için medyan ({med}) kullanıldı", flush=True)
    return np.array([val.get(c, med) for c in codes], dtype=int)


def synthetic(n_days=529, n_funds=60, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end="2026-09-25", periods=n_days)
    drift = rng.normal(0.0006, 0.0004, n_funds)
    vol = rng.uniform(0.003, 0.02, n_funds)
    r = rng.normal(drift, vol, (n_days, n_funds))
    px = 10 * np.exp(np.cumsum(r, axis=0))
    return pd.DataFrame(px, index=dates, columns=[f"F{i:02d}" for i in range(n_funds)])


# ----------------------------------------------------------------- skor bileşenleri
def _wsz(x, k):
    ok = ~np.isnan(x)
    out = np.zeros_like(x)
    if ok.sum() < 5:
        return out
    lo, hi = np.percentile(x[ok], [1, 99])
    v = np.clip(x[ok], lo, hi)
    sd = v.std(ddof=1)
    z = (v - v.mean()) / (sd if sd > 1e-12 else 1.0)
    out[ok] = 1.0 / (1.0 + np.exp(-k * z))
    return out


def build_components(px, k, sortino_gun, rf_annual):
    """C[t, fon, 7]: 6 sigmoid'li dönemsel + sortino (0-1 rank). Sadece t-252 ve sonrası anlamlı."""
    P = px.values
    T, N = P.shape
    C = np.zeros((T, N, NCOMP))
    for j, L in enumerate(LOOKBACKS):
        ret = np.full((T, N), np.nan)
        ret[L:] = (P[L:] / P[:-L] - 1.0) * 100.0
        for t in range(T):
            C[t, :, j] = _wsz(ret[t], k)
    lr = pd.DataFrame(np.log(P)).diff()
    W = sortino_gun
    mar = rf_annual / 252.0
    mean = lr.rolling(W, min_periods=W).mean()
    dd = np.sqrt((lr.clip(upper=0) ** 2).rolling(W, min_periods=W).mean())
    sortino = ((mean - mar) / dd.clip(lower=1e-4) * np.sqrt(W)).values
    for t in range(T):
        row = sortino[t]
        ok = ~np.isnan(row)
        if ok.sum() >= 5:
            rk = rankdata(row[ok])
            C[t, ok, 6] = (rk - 1) / (ok.sum() - 1)
    R = np.zeros((T, N))
    R[1:] = np.nan_to_num(P[1:] / P[:-1] - 1.0)
    return C, R


# ----------------------------------------------------------------- simülasyon
def simulate(S, R, valor, tradable, start, end, P_arr, X_arr):
    """Gerçekçi tek-portföy simülasyonu. S: (T,N) skor. Dönüş: (getiri, işlem sayısı)."""
    N = S.shape[1]
    pos = {}        # fon -> [değer, giriş_günü]
    sell_at = {}    # çıkış günü -> [fonlar]
    buy_at = {}     # giriş günü -> [(fon, tutar)]
    chunks = []     # [(kullanılabilir_gün, tutar)]
    idle = 1.0
    sell_pending = set()
    trades = 0
    for d in range(start, end):
        for f, pv in pos.items():
            if pv[1] < d:
                pv[0] *= 1.0 + R[d, f]
        for f in sell_at.pop(d, []):
            chunks.append((d + int(valor[f]) + 1, pos.pop(f)[0]))
            sell_pending.discard(f)
            trades += 1
        for f, amt in buy_at.pop(d, []):
            pos[f] = [amt, d]
            trades += 1
        if chunks:
            keep = []
            for av, amt in chunks:
                if av <= d:
                    idle += amt
                else:
                    keep.append((av, amt))
            chunks = keep
        if d + 1 >= end:
            continue
        row = np.where(tradable, S[d], -np.inf)
        Pn, Xn = int(P_arr[d]), max(int(X_arr[d]), int(P_arr[d]))
        for f in list(pos.keys()):
            if f in sell_pending:
                continue
            if 1 + int((row > row[f]).sum()) > Xn:
                sell_at.setdefault(d + 1, []).append(f)
                sell_pending.add(f)
        pend_buys = sum(len(v) for v in buy_at.values())
        active = len(pos) - len(sell_pending)
        outstanding = len(sell_pending) + len(chunks)
        n_buy = Pn - (active + pend_buys + outstanding)
        if n_buy > 0 and idle > 1e-12:
            blocked = set(pos.keys()) | {f for v in buy_at.values() for f, _ in v}
            order = np.argsort(-row)
            picks = []
            for f in order:
                if not np.isfinite(row[f]):
                    break
                if f not in blocked:
                    picks.append(int(f))
                    if len(picks) == n_buy:
                        break
            if picks:
                amt = idle / len(picks)
                for f in picks:
                    buy_at.setdefault(d + 1, []).append((f, amt))
                idle = 0.0
    wealth = idle + sum(a for _, a in chunks) + sum(v[0] for v in pos.values()) \
        + sum(a for v in buy_at.values() for _, a in v)
    return wealth - 1.0, trades


def _theta_to_arrays(theta):
    w = np.array(theta[:7], dtype=float)
    return w, int(round(theta[7])), int(round(theta[8]))


def _objective(theta):
    g = _G
    w, Pn, Xn = _theta_to_arrays(theta)
    s0, e0 = g["start"], g["end"]
    S = np.zeros((g["C"].shape[0], g["C"].shape[1]))
    S[s0:e0] = g["C"][s0:e0] @ w
    Pa = np.full(S.shape[0], Pn)
    Xa = np.full(S.shape[0], Xn)
    ret, _ = simulate(S, g["R"], g["valor"], g["tradable"], s0, e0, Pa, Xa)
    return -ret


def optimize(C, R, valor, tradable, start, end, a):
    _G.update(C=C, R=R, valor=valor, tradable=tradable, start=start, end=end)
    bounds = [(0.05, 6.0)] * 7 + [(2, 4), (2, 12)]
    integ = np.array([False] * 7 + [True, True])
    res = differential_evolution(
        _objective, bounds, integrality=integ, popsize=a.popsize, maxiter=a.maxiter,
        tol=0.01, seed=a.seed, polish=False, workers=a.workers,
        updating="deferred" if a.workers != 1 else "immediate", init="latinhypercube")
    return res.x, -res.fun


def run_fixed(C, R, valor, tradable, start, end, theta):
    w, Pn, Xn = _theta_to_arrays(theta)
    S = np.zeros(C.shape[:2])
    S[start:end] = C[start:end] @ w
    n = C.shape[0]
    return simulate(S, R, valor, tradable, start, end, np.full(n, Pn), np.full(n, Xn))


def run_walkforward(C, R, valor, tradable, t0, t1, train, a):
    n, N = C.shape[0], C.shape[1]
    S = np.zeros((n, N))
    Pa, Xa = np.full(n, 2), np.full(n, 8)
    hist, cur, tt = [], None, time.time()
    for d in range(t0, t1):
        if (d - t0) % a.reopt_every == 0:
            cur, _ = optimize(C, R, valor, tradable, d - train, d, a)
            hist.append(cur)
            k = len(hist)
            tot = -(-(t1 - t0) // a.reopt_every)
            el = time.time() - tt
            print(f"    [wf] {k}/{tot} yeniden optimizasyon, geçen {el/60:.1f} dk, kalan ~{el/k*(tot-k)/60:.1f} dk", flush=True)
        w, Pn, Xn = _theta_to_arrays(cur)
        S[d] = C[d] @ w
        Pa[d], Xa[d] = Pn, Xn
    ret, tr = simulate(S, R, valor, tradable, t0, t1, Pa, Xa)
    return ret, tr, np.array(hist)


# ----------------------------------------------------------------- ana akış
def fit_windows(usable, train, test, min_test):
    """Veri yetmezse önce test'i, gerekirse train'i küçült; uyarı metni döndür."""
    if train + test <= usable:
        return train, test, None
    t2 = usable - train
    if t2 >= min_test:
        return train, t2, f"TEST {test} -> {t2} küçültüldü (kullanılabilir gün: {usable})"
    tr2 = usable - min_test
    return tr2, min_test, f"TRAIN {train} -> {tr2}, TEST {test} -> {min_test} küçültüldü (kullanılabilir gün: {usable})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="price_history.json")
    ap.add_argument("--valor", default="fon_listesi.json")
    ap.add_argument("--train", type=int, default=252)
    ap.add_argument("--test", type=int, default=252)
    ap.add_argument("--min-test", type=int, default=60)
    ap.add_argument("--n-windows", type=int, default=1)
    ap.add_argument("--window-step", type=int, default=21)
    ap.add_argument("--reopt-every", type=int, default=1, help="walk-forward yeniden optimizasyon sıklığı (gün)")
    ap.add_argument("--popsize", type=int, default=12)
    ap.add_argument("--maxiter", type=int, default=30)
    ap.add_argument("--workers", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=float, default=3.0)
    ap.add_argument("--sortino-gun", type=int, default=69)
    ap.add_argument("--rf", type=float, default=0.42, help="yıllık risksiz getiri")
    ap.add_argument("--baseline-sortino", type=float, default=1.5)
    ap.add_argument("--baseline-p", type=int, default=2)
    ap.add_argument("--baseline-x", type=int, default=9)
    ap.add_argument("--exclude", default="BBR,END,BRG,YVD")
    ap.add_argument("--out", default="sonuclar")
    ap.add_argument("--no-wf", action="store_true")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()

    if a.inspect:
        for p in (a.prices, a.valor):
            with open(p, encoding="utf-8") as f:
                o = json.load(f)
            print(p, type(o).__name__, "len", len(o))
            s = next(iter(o.items())) if isinstance(o, dict) else o[0]
            print("  örnek:", str(s)[:400])
        return

    if a.synthetic:
        px = synthetic()
        valor = np.random.default_rng(1).integers(0, 5, px.shape[1])
    else:
        px = load_prices(a.prices)
        valor = load_valor(a.valor, list(px.columns))
    T, N = px.shape
    print(f"Veri: {T} gün x {N} fon, {px.index[0].date()} -> {px.index[-1].date()}", flush=True)

    excl = {c.strip() for c in a.exclude.split(",") if c.strip()}
    tradable = np.array([c not in excl for c in px.columns])
    t_min = max(LOOKBACKS + [a.sortino_gun])
    usable = T - t_min
    train, test, warn = fit_windows(usable, a.train, a.test, a.min_test)
    print(f"İstenen TRAIN/TEST = {a.train}/{a.test}; kullanılan = {train}/{test}", flush=True)
    if warn:
        print(f"[UYARI] Veri derinliği yetersiz (en uzun geri bakış {t_min} gün): {warn}", flush=True)

    print("Bileşenler hesaplanıyor...", flush=True)
    C, R = build_components(px, a.k, a.sortino_gun, a.rf)
    base_theta = BASE_COEFS + [a.baseline_sortino, a.baseline_p, a.baseline_x]

    results = []
    for wi in range(a.n_windows):
        t1 = T - wi * a.window_step
        t0 = t1 - test
        ts = t0 - train
        if ts < t_min:
            print(f"[bilgi] pencere {wi+1} için yeterli veri yok, duruldu", flush=True)
            break
        print(f"\n=== Pencere {wi+1}: train [{px.index[ts].date()}..{px.index[t0-1].date()}] "
              f"test [{px.index[t0].date()}..{px.index[t1-1].date()}] ===", flush=True)
        b_ret, b_tr = run_fixed(C, R, valor, tradable, t0, t1, base_theta)
        b_train, _ = run_fixed(C, R, valor, tradable, ts, t0, base_theta)
        t = time.time()
        th, tr_ret = optimize(C, R, valor, tradable, ts, t0, a)
        s_ret, s_tr = run_fixed(C, R, valor, tradable, t0, t1, th)
        print(f"  baseline: train %{b_train*100:.2f}  test %{b_ret*100:.2f}", flush=True)
        print(f"  DE statik: train %{tr_ret*100:.2f}  test %{s_ret*100:.2f}  ({time.time()-t:.0f} sn)", flush=True)
        print("  DE statik katsayılar:", dict(zip(COEF_NAMES + ["sortino", "P", "top_x"],
                                                [round(float(x), 2) for x in th[:7]] + [int(round(th[7])), int(round(th[8]))])), flush=True)
        row = dict(pencere=wi + 1, train_gun=train, test_gun=test,
                   baseline_train=b_train, baseline_test=b_ret, de_static_train=tr_ret, de_static_test=s_ret,
                   de_static_params=[float(x) for x in th])
        if not a.no_wf:
            w_ret, w_tr, hist = run_walkforward(C, R, valor, tradable, t0, t1, train, a)
            print(f"  DE günlük walk-forward test: %{w_ret*100:.2f}  (işlem {w_tr})", flush=True)
            row.update(de_wf_test=w_ret, de_wf_mean_params=[float(x) for x in hist.mean(axis=0)])
        results.append(row)

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "sonuc.json"), "w", encoding="utf-8") as f:
        json.dump(dict(train_istenen=a.train, test_istenen=a.test, train=train, test=test,
                       uyari=warn, sonuclar=results), f, ensure_ascii=False, indent=2)
    lines = ["pencere | baseline | DE statik | DE walk-forward (test getirisi %)"]
    for r in results:
        lines.append(f"{r['pencere']} | {r['baseline_test']*100:.2f} | {r['de_static_test']*100:.2f} | "
                     f"{r.get('de_wf_test', float('nan'))*100:.2f}")
    with open(os.path.join(a.out, "ozet.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
