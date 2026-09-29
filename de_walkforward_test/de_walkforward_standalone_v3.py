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
  python de_walkforward_standalone.py --sim-n 300 --sim-band 0.30 --train 150 --test 45 --n-windows 5 --window-step 20 --sortino-gun 138
      # DE yok: baseline etrafinda rastgele katsayi setleri, ayni pencerelerde sadece simulasyon (hassasiyet olcumu)
  python de_walkforward_standalone_v3.py --bench-n 500 --train 150 --test 45 --n-windows 5 --window-step 20 --sortino-gun 138
      # DE yok: null/benchmark. Baseline'i rastgele skorlar ve tek metrikli skorlarla ayni omurgada karsilastirir
"""
import argparse
import json
import os
import multiprocessing as mp
import pickle
import sys
import time
from functools import partial

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import rankdata

LOOKBACKS = [1, 5, 21, 63, 126, 252]
COEF_NAMES = ["gun", "hafta", "ay", "3ay", "6ay", "yil"]
BASE_COEFS = [0.43, 0.67, 2.41, 2.25, 1.55, 2.53]
SORTINO_MODU = "sigmoid"  # "sigmoid" (üretimle uyumlu) veya "rank" (eski)
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


def find_file(name, here):
    if os.path.exists(name):
        return name
    roots = [os.getcwd(), here, os.path.dirname(here), os.path.dirname(os.path.dirname(here))]
    for r in dict.fromkeys(roots):
        for dp, dn, fn in os.walk(r):
            dn[:] = [d for d in dn if d not in (".git", "node_modules", "__pycache__")]
            if name in fn:
                return os.path.join(dp, name)
    return name


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
    ok = np.isfinite(x)
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
        if SORTINO_MODU == "sigmoid":   # üretim: winsorize -> z-skor -> sigmoid (puanlama_motoru ile aynı)
            C[t, :, 6] = _wsz(sortino[t], k)
        else:                            # eski davranış: 0-1 yüzdelik sıra
            row = sortino[t]
            ok = ~np.isnan(row)
            if ok.sum() >= 5:
                rk = rankdata(row[ok])
                C[t, ok, 6] = (rk - 1) / (ok.sum() - 1)
    R = np.zeros((T, N))
    R[1:] = np.nan_to_num(P[1:] / P[:-1] - 1.0)
    return C, R


# ----------------------------------------------------------------- simülasyon
def simulate(S, R, valor, tradable, start, end, P_arr, X_arr, log=None):
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
            pv_ = pos.pop(f)
            if log is not None:
                log.append((f, pv_[1], d, pv_[0] / pv_[2] - 1.0, ""))
            chunks.append((d + int(valor[f]) + 1, pv_[0]))
            sell_pending.discard(f)
            trades += 1
        for f, amt in buy_at.pop(d, []):
            pos[f] = [amt, d, amt]
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
    if log is not None:
        for f, pv_ in pos.items():
            log.append((f, pv_[1], end, pv_[0] / pv_[2] - 1.0, "açık"))
    wealth = idle + sum(a for _, a in chunks) + sum(v[0] for v in pos.values()) \
        + sum(a for v in buy_at.values() for _, a in v)
    return wealth - 1.0, trades


def _theta_to_arrays(theta):
    w = np.array(theta[:7], dtype=float)
    return w, int(round(theta[7])), int(round(theta[8]))


def _init_worker(C, R, valor, tradable):
    _G.update(C=C, R=R, valor=valor, tradable=tradable)


def _objective_se(s0, e0, theta):
    g = _G
    w, Pn, Xn = _theta_to_arrays(theta)
    S = np.zeros((g["C"].shape[0], g["C"].shape[1]))
    S[s0:e0] = g["C"][s0:e0] @ w
    Pa = np.full(S.shape[0], Pn)
    Xa = np.full(S.shape[0], Xn)
    ret, _ = simulate(S, g["R"], g["valor"], g["tradable"], s0, e0, Pa, Xa)
    return -ret


def optimize(C, R, valor, tradable, start, end, a):
    func = partial(_objective_se, start, end)
    pool = a._pool
    bounds = [(0.05, 6.0)] * 7 + [(2, 4), (2, 12)]
    integ = np.array([False] * 7 + [True, True])
    res = differential_evolution(
        func, bounds, integrality=integ, popsize=a.popsize, maxiter=a.maxiter,
        tol=0.01, seed=a.seed, polish=False,
        workers=pool.map if pool is not None else 1,
        updating="deferred" if pool is not None else "immediate", init="latinhypercube")
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


class _Tee:
    def __init__(self, *fs):
        self.fs = fs

    def write(self, x):
        for f in self.fs:
            f.write(x)

    def flush(self):
        for f in self.fs:
            f.flush()


# ----------------------------------------------------------------- baseline etrafında hassasiyet simülasyonu
def _sim_task(task):
    s0, e0, theta = task
    return -_objective_se(s0, e0, theta)


def _pct_rank(dist, x):
    """x'in dağılımdaki yüzdelik konumu (0-100): dağılımın x'ten küçük olan payı (eşitler yarım sayılır)."""
    dist = np.asarray(dist)
    return 100.0 * ((dist < x).sum() + 0.5 * (dist == x).sum()) / len(dist)


def run_sim(a, windows, px, base_theta, base2, C, R, valor, tradable, T):
    """windows: [(ts, t0, t1)]. Optimizasyon yok; her set için baseline katsayıları x U(1-band, 1+band).
    Aynı setler tüm pencerelerde kullanılır (tohum sabit), fark yalnız katsayılardan gelir."""
    rng = np.random.default_rng(a.sim_seed)
    base = np.array(base_theta, dtype=float)
    sets = np.tile(base, (a.sim_n, 1))
    sets[:, :6] = base[:6] * rng.uniform(1 - a.sim_band, 1 + a.sim_band, (a.sim_n, 6))
    if a.sim_vary_sortino:
        sets[:, 6] = base[6] * rng.uniform(1 - a.sim_band, 1 + a.sim_band, a.sim_n)
    nw = len(windows)
    tasks = []
    for (ts, t0, t1) in windows:
        for th in sets:
            tasks.append((t0, t1, th))   # test
        for th in sets:
            tasks.append((ts, t0, th))   # train
    print(f"\nSimülasyon: {a.sim_n} set x {nw} pencere x (train+test) = {len(tasks)} koşu, band ±%{a.sim_band*100:.0f}"
          f"{', sortino ağırlığı da oynatıldı' if a.sim_vary_sortino else ''}", flush=True)
    t = time.time()
    if a._pool is not None:
        out = a._pool.map(_sim_task, tasks, chunksize=max(1, len(tasks) // (8 * (os.cpu_count() or 1))))
    else:
        out = [_sim_task(x) for x in tasks]
    print(f"Simülasyon bitti: {time.time()-t:.0f} sn", flush=True)
    out = np.array(out).reshape(nw, 2, a.sim_n) * 100.0  # [pencere, test/train, set] (%)
    test, train = out[:, 0, :], out[:, 1, :]

    ref = {}
    for name, th in (("baseline", base_theta), ("cenker", base2)):
        te = np.array([run_fixed(C, R, valor, tradable, t0, t1, th)[0] for (_, t0, t1) in windows]) * 100.0
        tr = np.array([run_fixed(C, R, valor, tradable, ts, t0, th)[0] for (ts, t0, _) in windows]) * 100.0
        ref[name] = (te, tr)

    def block(title, mat, ref_idx):
        lines = [title]
        avg = mat.mean(axis=0)
        q = np.percentile(avg, [5, 50, 95])
        lines.append(f"  set ortalaması (pencereler üstü): ort {avg.mean():.2f}  std {avg.std(ddof=1):.2f}  "
                     f"%5 {q[0]:.2f}  %50 {q[1]:.2f}  %95 {q[2]:.2f}  min {avg.min():.2f}  max {avg.max():.2f}")
        for name in ("baseline", "cenker"):
            v = ref[name][ref_idx].mean()
            lines.append(f"  {name}: ort {v:.2f}  dağılımdaki yüzdelik {_pct_rank(avg, v):.0f}")
        lines.append("  pencere | ort | std | %5 | %50 | %95 | baseline (yüzdelik) | cenker (yüzdelik)")
        for w in range(nw):
            m = mat[w]
            qq = np.percentile(m, [5, 50, 95])
            b, c = ref["baseline"][ref_idx][w], ref["cenker"][ref_idx][w]
            lines.append(f"  {w+1} | {m.mean():.2f} | {m.std(ddof=1):.2f} | {qq[0]:.2f} | {qq[1]:.2f} | {qq[2]:.2f} | "
                         f"{b:.2f} (%{_pct_rank(m, b):.0f}) | {c:.2f} (%{_pct_rank(m, c):.0f})")
        return lines

    lines = block("TEST getirisi % (out-of-sample)", test, 0) + [""] + block("TRAIN getirisi % (aynı setler, referans)", train, 1)
    # baseline'ın train'de mi yoksa test'te mi 'seçilmiş' göründüğü tek bakışta
    lines += ["", "Not: yüzdelik = setlerin kaçta kaçı bu değerin ALTINDA kaldı. Test'te yüksek (>%80) ise ince ayar "
              "bir şey katıyor; ~%50 ise baseline sıradan bir nokta; std küçükse etrafında dolaşmak güvenli."]
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "sim_ozet.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(a.out, "sim_sonuc.json"), "w", encoding="utf-8") as f:
        json.dump(dict(band=a.sim_band, n=a.sim_n, seed=a.sim_seed, vary_sortino=a.sim_vary_sortino,
                       base_theta=[float(x) for x in base_theta], base2=base2,
                       sets=sets.tolist(), test_pct=test.tolist(), train_pct=train.tolist(),
                       baseline_test=ref["baseline"][0].tolist(), cenker_test=ref["cenker"][0].tolist(),
                       baseline_train=ref["baseline"][1].tolist(), cenker_train=ref["cenker"][1].tolist()),
                  f, ensure_ascii=False)
    print("\n" + "\n".join(lines), flush=True)


# ----------------------------------------------------------------- null / benchmark karşılaştırması
def _rnd_scores(seed, rho, T, N):
    """Rastgele skor matrisi. rho=0: her gün bağımsız (aşırı devir); rho=1: sabit (al-tut);
    arası: AR(1), skorlar günden güne kalıcı (gerçek skorlara daha yakın devir)."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((T, N))
    if rho <= 0:
        return e
    if rho >= 1:
        return np.tile(e[0], (T, 1))
    S = np.empty_like(e)
    S[0] = e[0]
    c = np.sqrt(1.0 - rho * rho)
    for t in range(1, T):
        S[t] = rho * S[t - 1] + c * e[t]
    return S


def _bench_task(task):
    kind, arg, s0, e0, Pn, Xn = task
    g = _G
    T, N = g["C"].shape[:2]
    if kind == "w":
        S = np.zeros((T, N))
        S[s0:e0] = g["C"][s0:e0] @ np.array(arg, dtype=float)
    else:
        S = _rnd_scores(arg[0], arg[1], T, N)
    ret, tr = simulate(S, g["R"], g["valor"], g["tradable"], s0, e0, np.full(T, Pn), np.full(T, Xn))
    return ret, tr


def run_bench(a, windows, base_theta, base2, C, R, valor, tradable, T, px=None):
    nw = len(windows)
    Pn, Xn = int(base_theta[7]), int(base_theta[8])
    rhos = [float(x) for x in a.bench_rho.split(",")]
    wb = list(base_theta[:7])
    w2 = list(base2[:7])
    det = []   # (isim, ağırlıklar, P, X)
    for j, nm in enumerate(COEF_NAMES + ["sortino"]):
        w = [0.0] * 7
        w[j] = 1.0
        det.append((f"tek: {nm}", w, Pn, Xn))
    det.append(("eşit 6 dönemsel", [1.0] * 6 + [0.0], Pn, Xn))
    det.append(("eşit 6 dönemsel + sortino", [1.0] * 7, Pn, Xn))
    det.append(("BASELINE (çok metrikli)", wb, Pn, Xn))
    det.append(("cenker katsayıları, aynı kural", w2, Pn, Xn))
    det.append((f"cenker, kendi kuralı (P={int(base2[7])}, X={int(base2[8])})", w2, int(base2[7]), int(base2[8])))

    tasks = []
    for (_, w, P_, X_) in det:
        for (ts, t0, t1) in windows:
            tasks.append(("w", tuple(w), t0, t1, P_, X_))     # test
            tasks.append(("w", tuple(w), ts, t0, P_, X_))     # train
    for rho in rhos:
        for i in range(a.bench_n):
            for (ts, t0, t1) in windows:
                tasks.append(("r", (a.bench_seed + i, rho), t0, t1, Pn, Xn))
                tasks.append(("r", (a.bench_seed + i, rho), ts, t0, Pn, Xn))
    print(f"\nBenchmark: {len(det)} deterministik puanlayıcı + {len(rhos)} rastgele null x {a.bench_n} tohum, "
          f"{nw} pencere, kural P={Pn} X={Xn} = {len(tasks)} koşu", flush=True)
    t = time.time()
    if a._pool is not None:
        out = a._pool.map(_bench_task, tasks, chunksize=max(1, len(tasks) // (8 * (os.cpu_count() or 1))))
    else:
        out = [_bench_task(x) for x in tasks]
    print(f"Benchmark bitti: {time.time()-t:.0f} sn", flush=True)
    ret = np.array([o[0] for o in out]) * 100.0
    trd = np.array([o[1] for o in out], dtype=float)
    nd = len(det) * nw * 2
    D = ret[:nd].reshape(len(det), nw, 2)          # [puanlayıcı, pencere, test/train]
    Dt = trd[:nd].reshape(len(det), nw, 2)
    Rn = ret[nd:].reshape(len(rhos), a.bench_n, nw, 2)
    Rt = trd[nd:].reshape(len(rhos), a.bench_n, nw, 2)
    r_test = Rn[..., 0].mean(axis=2)                # [rho, tohum] pencereler üstü
    r_train = Rn[..., 1].mean(axis=2)

    L = [f"BENCHMARK (aynı omurga: valör/T+1, P={Pn}, top_x={Xn}; {nw} pencere; sortino {a.sortino_gun} gün; sortino modu {a.sortino_modu})", ""]
    L.append(f"Rastgele null'lar (skor = rastgele; rho = günden güne kalıcılık; {a.bench_n} tohum)")
    L.append("  rho | işlem/pencere | TEST ort | std | %5 | %50 | %95 | TRAIN ort | aşırı (ort>%200, ort/std dışı bırakıldı)")
    for k, rho in enumerate(rhos):
        q = np.percentile(r_test[k], [5, 50, 95])
        ok_te, ok_tr = r_test[k] < 200, r_train[k] < 200
        nx = int((~ok_te).sum()), int((~ok_tr).sum())
        L.append(f"  {rho:.2f} | {Rt[k, ..., 0].mean():.1f} | {r_test[k][ok_te].mean():.2f} | {r_test[k][ok_te].std(ddof=1):.2f} | "
                 f"{q[0]:.2f} | {q[1]:.2f} | {q[2]:.2f} | {r_train[k][ok_tr].mean():.2f} | test {nx[0]}, train {nx[1]}")
    L.append("")
    L.append("Puanlayıcılar: TEST % (pencere ortalaması) | işlem/pencere | TRAIN % | pencereler (test) | TEST yüzdeliği: " +
             " ".join(f"rho{r:.2f}" for r in rhos))
    for i, (nm, _, _, _) in enumerate(det):
        te = D[i, :, 0]
        pcts = " ".join(f"%{_pct_rank(r_test[k], te.mean()):.0f}" for k in range(len(rhos)))
        L.append(f"  {nm} | {te.mean():.2f} | {Dt[i, :, 0].mean():.1f} | {D[i, :, 1].mean():.2f} | "
                 f"[{', '.join(f'{x:.1f}' for x in te)}] | {pcts}")
    L.append("")
    ib = next(i for i, d in enumerate(det) if d[0].startswith("BASELINE"))
    L.append("BASELINE'ın pencere bazında yüzdeliği (test) her null'a göre:")
    for k, rho in enumerate(rhos):
        ps = [f"%{_pct_rank(Rn[k, :, w, 0], D[ib, w, 0]):.0f}" for w in range(nw)]
        L.append(f"  rho {rho:.2f}: " + " ".join(ps))
    singles = [i for i, d in enumerate(det) if d[0].startswith("tek:")]
    best_tr = max(singles, key=lambda i: D[i, :, 1].mean())
    best_te = max(singles, key=lambda i: D[i, :, 0].mean())
    L.append("")
    L.append(f"Train'de en iyi tek metrik: {det[best_tr][0]} (train {D[best_tr, :, 1].mean():.2f}, test {D[best_tr, :, 0].mean():.2f})")
    L.append(f"Test'te en iyi tek metrik (seçim test'e bakılarak yapıldı, iyimser): {det[best_te][0]} (test {D[best_te, :, 0].mean():.2f})")
    L.append(f"Tek metriklerin test ortalaması: {np.mean([D[i, :, 0].mean() for i in singles]):.2f};  BASELINE test: {D[ib, :, 0].mean():.2f}")
    L.append("")
    L.append("Okuma: BASELINE rastgele null'ların belirgin üstünde (>%90) VE tek metriklerin en iyisinden iyiyse, çok metrikli birleşimin değeri var. "
             "Tek metrikle aynı seviyedeyse karmaşıklığın getirisi yok. Null'ların içindeyse getiri büyük ölçüde rejimden. "
             "rho=0 (her gün yeni rastgele) aşırı devir yapar, adil olmayan zayıf bir null'dır; rho=1 al-tut rastgele fonlardır; "
             "işlem sayısı baseline'a yakın rho en anlamlı karşılaştırmadır.")
    if a.bench_detay and px is not None:
        DL = ["Test pencerelerinde alınan pozisyonlar: fon | giriş | çıkış | pozisyon getirisi %  (çıkış 'açık' = pencere sonunda hâlâ elde)"]
        for (nm, w, P_, X_) in det:
            for wi, (ts, t0, t1) in enumerate(windows):
                log = []
                S_ = np.zeros((T, len(px.columns)))
                S_[t0:t1] = C[t0:t1] @ np.array(w, dtype=float)
                rr, _ = simulate(S_, R, valor, tradable, t0, t1, np.full(T, P_), np.full(T, X_), log=log)
                DL.append(f"[{nm}] pencere {wi+1}: test %{rr*100:.2f}")
                for (f_, e_, x_, g_, op_) in sorted(log, key=lambda z: z[1]):
                    flag = "  <-- >%30" if abs(g_) > 0.30 else ""
                    DL.append(f"    {px.columns[f_]} | {px.index[e_].date()} | {px.index[min(x_, T-1)].date()} {op_} | %{g_*100:.1f}{flag}")
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "bench_islemler.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(DL) + "\n")
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "bench_ozet.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(a.out, "bench_sonuc.json"), "w", encoding="utf-8") as f:
        json.dump(dict(P=Pn, X=Xn, rhos=rhos, n=a.bench_n, seed=a.bench_seed,
                       puanlayicilar=[d[0] for d in det], det_pct=D.tolist(), det_islem=Dt.tolist(),
                       null_test_pct=Rn[..., 0].tolist(), null_train_pct=Rn[..., 1].tolist()),
                  f, ensure_ascii=False)
    print("\n" + "\n".join(L), flush=True)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    logf = open(os.path.join(here, "de_wf_full_log.txt"), "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, logf)
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
    ap.add_argument("--baseline2", default="3.2,6.3,19.9,28.7,14.8,24.6,2.5,2,10",
                    help="ikinci baseline (Cenker Serenity): 6 donemsel, sortino agirligi, P, top_x")
    ap.add_argument("--exclude", default="BBR,END,BRG,YVD")
    ap.add_argument("--out", default=None, help="varsayilan: script klasoru")
    ap.add_argument("--no-wf", action="store_true")
    ap.add_argument("--sim-n", type=int, default=0, help=">0 ise DE/walk-forward yerine baseline etrafinda N rastgele set simule et")
    ap.add_argument("--sim-band", type=float, default=0.30, help="katsayi sapma bandi (0.30 = x U(0.7,1.3))")
    ap.add_argument("--sim-seed", type=int, default=7)
    ap.add_argument("--sim-vary-sortino", action="store_true", help="sortino agirligini da ayni bantta oynat")
    ap.add_argument("--sortino-modu", choices=["sigmoid", "rank"], default="sigmoid",
                    help="sortino bileseni: sigmoid (uretimle ayni, varsayilan) veya rank (eski 0-1 siralama)")
    ap.add_argument("--bench-n", type=int, default=0, help=">0 ise null/benchmark modu: rastgele skor tohum sayisi")
    ap.add_argument("--bench-rho", default="0,0.95,1", help="rastgele skorlarin gunluk kalicilik degerleri (virgulle)")
    ap.add_argument("--bench-seed", type=int, default=1000)
    ap.add_argument("--bench-detay", action="store_true", help="benchmark'ta her puanlayicinin test pencerelerindeki pozisyonlarini bench_islemler.txt'ye yaz")
    ap.add_argument("--veri-kontrol", action="store_true", help="fiyat verisinde asiri gunluk hareketleri listele ve cik")
    ap.add_argument("--veri-esik", type=float, default=0.30, help="veri kontrolunde gunluk |getiri| esigi (0.30 = %30)")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    global SORTINO_MODU
    SORTINO_MODU = a.sortino_modu

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
        a.prices = find_file(a.prices, here)
        a.valor = find_file(a.valor, here)
        print(f"Fiyat dosyasi: {a.prices}\nValor dosyasi: {a.valor}", flush=True)
        px = load_prices(a.prices)
        px = px.where(px > 0).ffill()
        px = px.loc[:, px.notna().sum() > 30]
        valor = load_valor(a.valor, list(px.columns))
    T, N = px.shape
    print(f"Veri: {T} gün x {N} fon, {px.index[0].date()} -> {px.index[-1].date()}", flush=True)

    if a.veri_kontrol:
        rr_ = px / px.shift(1) - 1.0
        st = rr_.stack()
        big = st[st.abs() > a.veri_esik]
        print(f"\nVeri kontrolü: |günlük getiri| > %{a.veri_esik*100:.0f} olan {len(big)} gözlem ({big.index.get_level_values(1).nunique()} fon)", flush=True)
        for (dt, cd), v in big.reindex(big.abs().sort_values(ascending=False).index).head(40).items():
            i_ = px.index.get_loc(dt)
            print(f"  {cd} | {dt.date()} | fiyat {px[cd].iloc[i_-1]:.6g} -> {px[cd].iloc[i_]:.6g} | %{v*100:.1f}", flush=True)
        print("  en düşük fiyatlı 10 fon (min fiyat):", flush=True)
        for cd, v in px.min().sort_values().head(10).items():
            print(f"    {cd} | {v:.6g}", flush=True)
        return

    if a.out is None:
        a.out = here
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
    nw = (os.cpu_count() or 1) if a.workers == -1 else a.workers
    _init_worker(C, R, valor, tradable)
    a._pool = mp.Pool(nw, initializer=_init_worker, initargs=(C, R, valor, tradable)) if nw > 1 else None
    print(f"İşçi süreç sayısı: {nw}", flush=True)
    base_theta = BASE_COEFS + [a.baseline_sortino, a.baseline_p, a.baseline_x]
    base2 = [float(x) for x in a.baseline2.split(",")]

    if a.sim_n > 0 or a.bench_n > 0:
        windows = []
        for wi in range(a.n_windows):
            t1 = T - wi * a.window_step
            t0 = t1 - test
            ts = t0 - train
            if ts < t_min:
                print(f"[bilgi] pencere {wi+1} için yeterli veri yok, duruldu", flush=True)
                break
            windows.append((ts, t0, t1))
            print(f"Pencere {wi+1}: train [{px.index[ts].date()}..{px.index[t0-1].date()}] "
                  f"test [{px.index[t0].date()}..{px.index[t1-1].date()}]", flush=True)
        if a.bench_n > 0:
            run_bench(a, windows, base_theta, base2, C, R, valor, tradable, T, px)
        else:
            run_sim(a, windows, px, base_theta, base2, C, R, valor, tradable, T)
        if a._pool is not None:
            a._pool.close()
        return

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
        b2_ret, _ = run_fixed(C, R, valor, tradable, t0, t1, base2)
        b2_train, _ = run_fixed(C, R, valor, tradable, ts, t0, base2)
        print(f"  baseline2 (Cenker): train %{b2_train*100:.2f}  test %{b2_ret*100:.2f}", flush=True)
        t = time.time()
        th, tr_ret = optimize(C, R, valor, tradable, ts, t0, a)
        s_ret, s_tr = run_fixed(C, R, valor, tradable, t0, t1, th)
        print(f"  baseline: train %{b_train*100:.2f}  test %{b_ret*100:.2f}", flush=True)
        print(f"  DE statik: train %{tr_ret*100:.2f}  test %{s_ret*100:.2f}  ({time.time()-t:.0f} sn)", flush=True)
        print("  DE statik katsayılar:", dict(zip(COEF_NAMES + ["sortino", "P", "top_x"],
                                                [round(float(x), 2) for x in th[:7]] + [int(round(th[7])), int(round(th[8]))])), flush=True)
        row = dict(pencere=wi + 1, train_gun=train, test_gun=test,
                   baseline_train=b_train, baseline_test=b_ret, baseline2_train=b2_train, baseline2_test=b2_ret, de_static_train=tr_ret, de_static_test=s_ret,
                   de_static_params=[float(x) for x in th])
        if not a.no_wf:
            w_ret, w_tr, hist = run_walkforward(C, R, valor, tradable, t0, t1, train, a)
            print(f"  DE günlük walk-forward test: %{w_ret*100:.2f}  (işlem {w_tr})", flush=True)
            row.update(de_wf_test=w_ret, de_wf_mean_params=[float(x) for x in hist.mean(axis=0)])
        results.append(row)
        with open(os.path.join(a.out, "de_wf_full_checkpoint.pkl"), "wb") as f:
            pickle.dump(results, f)

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "sonuc.json"), "w", encoding="utf-8") as f:
        json.dump(dict(train_istenen=a.train, test_istenen=a.test, train=train, test=test,
                       uyari=warn, sonuclar=results), f, ensure_ascii=False, indent=2)
    lines = ["pencere | baseline (senin) | baseline2 (Cenker) | DE statik | DE walk-forward (test getirisi %)"]
    for r in results:
        lines.append(f"{r['pencere']} | {r['baseline_test']*100:.2f} | {r['baseline2_test']*100:.2f} | {r['de_static_test']*100:.2f} | "
                     f"{r.get('de_wf_test', float('nan'))*100:.2f}")
    with open(os.path.join(a.out, "ozet.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
