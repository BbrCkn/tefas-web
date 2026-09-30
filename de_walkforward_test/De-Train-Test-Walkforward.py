"""
de_train_test_walkforward.py - Her pencerede DE'yi train'de optimize et, testte dene
Bu senin G maddendeki "DE baseline'ı geçemez" kanıtı için
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import math
from scipy.optimize import differential_evolution

KOK = Path(__file__).parent
VERI_YOLU = KOK / "price_history_6y.json"
RESULT_DIR = KOK / "results"
RESULT_DIR.mkdir(exist_ok=True)

BASELINE = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.5])

def yukle():
    with open(VERI_YOLU, encoding="utf-8") as f:
        data = json.load(f)
    tarihler_desc = data["tarihler"]
    fiyatlar = data["fiyatlar"]
    tarihler_asc = tarihler_desc[::-1]
    fiyatlar_asc = {kod: np.array(seri[::-1], dtype=float) for kod, seri in fiyatlar.items()}
    return tarihler_asc, fiyatlar_asc

def hesapla_donemsel_metrikler(fiyatlar_asc, gun_idx):
    offsets = [1, 5, 21, 63, 126, 252]
    kodlar = list(fiyatlar_asc.keys())
    mat = np.zeros((len(kodlar), 6))
    for j, off in enumerate(offsets):
        if gun_idx - off < 0:
            continue
        for i, kod in enumerate(kodlar):
            now = fiyatlar_asc[kod][gun_idx]
            past = fiyatlar_asc[kod][gun_idx - off]
            if past > 1e-9:
                mat[i, j] = (now / past - 1) * 100
    norm = np.zeros_like(mat)
    for j in range(6):
        col = mat[:, j]
        valid = col[col != 0]
        if len(valid) < 10:
            continue
        p1, p99 = np.percentile(valid, [1, 99])
        col = np.clip(col, p1, p99)
        mean, std = col.mean(), col.std()
        if std < 1e-9:
            std = 1e-9
        z = (col - mean) / std
        z = np.clip(z, -3, 3)
        sig = 1 / (1 + np.exp(-z))
        norm[:, j] = sig
    return kodlar, norm

def simulate(fiyatlar_asc, start_idx, end_idx, katsayi, periyod=2):
    kodlar = list(fiyatlar_asc.keys())
    _, norm0 = hesapla_donemsel_metrikler(fiyatlar_asc, start_idx)
    skor0 = norm0 @ katsayi
    top2_idx = np.argsort(skor0)[-2:]
    portfoy = [kodlar[i] for i in top2_idx if skor0[i] > 0]
    if len(portfoy) < 2:
        portfoy = [kodlar[i] for i in np.argsort(skor0)[-2:]]
    sermaye = [100.0]
    bekleyen = {}
    for gi in range(start_idx+1, end_idx+1):
        _, norm = hesapla_donemsel_metrikler(fiyatlar_asc, gi)
        skor = norm @ katsayi
        order = np.argsort(-skor)
        rank_dict = {kodlar[order[i]]: i+1 for i in range(len(kodlar))}
        for kod in list(portfoy):
            if kod in rank_dict and rank_dict[kod] > 10:
                bekleyen[kod] = gi + periyod
        for kod, sat in list(bekleyen.items()):
            if gi >= sat:
                if kod in portfoy:
                    portfoy.remove(kod)
                del bekleyen[kod]
                for idx in order:
                    y = kodlar[idx]
                    if y not in portfoy:
                        portfoy.append(y)
                        break
        g = 0; cnt = 0
        for kod in portfoy:
            if fiyatlar_asc[kod][gi-1] > 1e-9:
                g += (fiyatlar_asc[kod][gi] / fiyatlar_asc[kod][gi-1] - 1)
                cnt += 1
        if cnt:
            g /= cnt
            sermaye.append(sermaye[-1] * (1+g))
        else:
            sermaye.append(sermaye[-1])
    if len(sermaye) < 2:
        return 0.0
    toplam = sermaye[-1]/sermaye[0] - 1
    gun = end_idx - start_idx
    yillik = (1+toplam)**(252/max(gun,1)) - 1
    return yillik*100

def main():
    tarihler_asc, fiyatlar_asc = yukle()
    print(f"Veri: {tarihler_asc[0]} -> {tarihler_asc[-1]}, {len(tarihler_asc)} gun")
    usable_start = 252; train_len=252; test_len=126
    pencereler=[]; idx=usable_start+train_len
    while idx+test_len < len(tarihler_asc):
        train_s=idx-train_len; train_e=idx-1; test_s=idx; test_e=idx+test_len-1
        pencereler.append((train_s,train_e,test_s,test_e)); idx+=test_len
    print(f"{len(pencereler)} pencere")

    rows=[]
    for pi,(train_s,train_e,test_s,test_e) in enumerate(pencereler):
        print(f"\n=== Pencere {pi+1} Train {tarihler_asc[train_s]}->{tarihler_asc[train_e]} Test {tarihler_asc[test_s]}->{tarihler_asc[test_e]} ===")
        # baseline train/test
        base_train = simulate(fiyatlar_asc, train_s, train_e, BASELINE)
        base_test = simulate(fiyatlar_asc, test_s, test_e, BASELINE)

        # DE objective: train'i maximize et
        def objective(k):
            return -simulate(fiyatlar_asc, train_s, train_e, np.array(k))

        bounds = [(0.1, 6.0)]*6
        # hizli DE: maxiter 15, popsize 10 -> 900 eval
        result = differential_evolution(objective, bounds, maxiter=15, popsize=10, seed=pi, polish=False, workers=1, disp=False)
        best_k = result.x
        best_train = -result.fun
        best_test = simulate(fiyatlar_asc, test_s, test_e, best_k)

        print(f"Baseline train {base_train:.2f}% test {base_test:.2f}%")
        print(f"DE train {best_train:.2f}% test {best_test:.2f}% k={np.round(best_k,2)}")
        print(f"Fark test: DE - baseline = {best_test - base_test:.2f}%")

        rows.append({
            "pencere": pi+1,
            "train_bas": tarihler_asc[train_s], "train_bit": tarihler_asc[train_e],
            "test_bas": tarihler_asc[test_s], "test_bit": tarihler_asc[test_e],
            "baseline_train": base_train, "baseline_test": base_test,
            "de_train": best_train, "de_test": best_test,
            "de_test_minus_base": best_test - base_test,
            "de_k": ",".join(f"{x:.2f}" for x in best_k)
        })

    df=pd.DataFrame(rows)
    df.to_csv(RESULT_DIR / "de_train_test_walkforward.csv", index=False)
    print("\n" + df.to_string())
    print(f"\nOrtalama de_test - baseline: {df['de_test_minus_base'].mean():.2f}%")
    print(f"Median de_test - baseline: {df['de_test_minus_base'].median():.2f}%")
    print(f"DE baseline'i yenme orani testte: {(df['de_test'] > df['baseline_test']).mean():.2f}")

if __name__ == "__main__":
    main()
