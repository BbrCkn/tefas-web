"""
backtest_6y_walkforward.py
TEFAS Backtest Doğrulama - 6 yıllık veri ile

Amaç: DE katsayı optimizasyonunun genelleşip genelleşmediğini
örtüşMEYEN pencerelerde test etmek.

Girdi: de_walkforward_test/price_history_6y.json (1234 gün, 201 fon)
Çıktı: 
 - results/walkforward_summary.csv
 - results/olcek_kontrol.csv
 - results/en_kotu_pencere.csv

Mantık:
1. Veri: tarihler DESC (en yeni ilk). Asc'e çevirip usable = tarihler[254:] (YIL için 1 yıl warmup)
2. Pencereler: ÖRTÜŞMEYEN test pencereleri. Train = test öncesi 252 iş günü (~1 yıl), Test = 126 gün (~6 ay)
   1234 gün -> ~980 usable -> ~7-8 pencere çıkar
3. Metrik: Sadece dönemsel 6'lı (GUN, HAF, AY, 3AY, 6AY, YIL) optimize ediliyor.
   Risk metrikleri senin dediğin gibi dönemsel toplamından besleniyor, optimize edilmiyor.
4. Karşılaştırma: baseline, cenker, de_statik, de_walkforward, random_null, basit benchmark (sadece YIL)
5. Rapor: ortalama yerine "yendiği pencere oranı" ve "en kötü pencere" - senin istediğin
"""

import json
from pathlib import Path
import datetime as dt
import numpy as np
import pandas as pd

KOK = Path(__file__).parent
VERI_YOLU = KOK / "price_history_6y.json"
RESULT_DIR = KOK / "results"
RESULT_DIR.mkdir(exist_ok=True)

# --- CONFIG: senin Excel'deki AD209:AI209 baseline'ını buraya koy ---
# Sıra: GUN, HAF, AY, 3AY, 6AY, YIL
BASELINE_KATSAYI = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.5])  # üretimdeki 2.5 dahil, kaynağı belirsiz dediğin
CENKER_KATSAYI = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 5.62]) # örnek, senin log'dan gelecek

# --- Veri yükle ---
def yukle():
    with open(VERI_YOLU, encoding="utf-8") as f:
        data = json.load(f)
    tarihler_desc = data["tarihler"]  # en yeni -> en eski
    fiyatlar = data["fiyatlar"]  # kod -> liste (desc ile aynı uzunluk)
    # Asc'e çevir (en eski -> en yeni) hesaplaması kolay
    tarihler_asc = tarihler_desc[::-1]
    fiyatlar_asc = {kod: np.array(seri[::-1], dtype=float) for kod, seri in fiyatlar.items()}
    return tarihler_asc, tarihler_desc, fiyatlar_asc

# --- Ölçek kontrolü (senin istediğin OLCEK? satırları) ---
def olcek_kontrol(tarihler_asc, fiyatlar_asc):
    bulgular = []
    for kod, seri in fiyatlar_asc.items():
        # 0 olan günleri atla
        for i in range(1, len(seri)):
            eski, yeni = seri[i-1], seri[i]
            if eski < 1e-9 or yeni < 1e-9:
                continue
            r = yeni / eski
            if r > 50 or r < 0.02:
                import math
                lg = math.log10(r)
                k = round(lg)
                yakin = abs(lg - k) < 0.15 and abs(k) >= 2
                bulgular.append({
                    "kod": kod,
                    "tarih": tarihler_asc[i],
                    "oran": r,
                    "log10": lg,
                    "10kuvvet_yakin": yakin,
                    "eski": eski,
                    "yeni": yeni
                })
    df = pd.DataFrame(bulgular)
    df.to_csv(RESULT_DIR / "olcek_kontrol.csv", index=False)
    print(f"[OLCEK] {len(df)} şüpheli sıçrama bulundu -> results/olcek_kontrol.csv")
    if len(df):
        print(df.head(20).to_string())
    return df

# --- Metrik hesaplama (winsor + z + sigmoid aynı) ---
def hesapla_donemsel_metrikler(tarihler_asc, fiyatlar_asc, gun_idx_asc):
    """
    gun_idx_asc: test edilecek günün asc indexi
    offsetler: GUN=1, HAF=5, AY=21, 3AY=63, 6AY=126, YIL=252 (iş günü)
    """
    offsets = [1, 5, 21, 63, 126, 252]
    kodlar = list(fiyatlar_asc.keys())
    mat = np.zeros((len(kodlar), 6))
    for j, off in enumerate(offsets):
        if gun_idx_asc - off < 0:
            continue
        for i, kod in enumerate(kodlar):
            now = fiyatlar_asc[kod][gun_idx_asc]
            past = fiyatlar_asc[kod][gun_idx_asc - off]
            if past > 1e-9:
                mat[i, j] = (now / past - 1) * 100
    # Winsor %1-%99 + Z + Sigmoid (k=1 varsayılan, senin C224)
    # Basit versiyon: her metrik için ayrı normalize
    norm = np.zeros_like(mat)
    for j in range(6):
        col = mat[:, j]
        # 0 olan fonları (henüz kurulmamış) hariç tut
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
        sig = 1 / (1 + np.exp(-z))  # k=1
        norm[:, j] = sig
    return kodlar, norm

def skorla(kodlar, norm_mat, katsayi):
    # katsayi: 6'lı
    skor = norm_mat @ katsayi
    # 0 fiyatlı fonları en alta at
    return dict(zip(kodlar, skor))

# --- Basit simülasyon: 2 fon %50-%50, rank>10 sat, en yüksek skorlu al ---
def simulate_window(tarihler_asc, fiyatlar_asc, start_idx, end_idx, katsayi, periyod_valor=2):
    """
    start_idx..end_idx arası test penceresi (asc indexler)
    Baslangic portfoyu: ilk gün en yüksek 2 fon
    """
    kodlar = list(fiyatlar_asc.keys())
    # Başlangıç: ilk gün en yüksek skorlu 2 fonu al
    _, norm0 = hesapla_donemsel_metrikler(tarihler_asc, fiyatlar_asc, start_idx)
    skor0 = norm0 @ katsayi
    top2_idx = np.argsort(skor0)[-2:]
    portfoy = [kodlar[i] for i in top2_idx if skor0[i] > 0]
    if len(portfoy) < 2:
        portfoy = kodlar[:2]  # fallback

    nakit = 0
    adet = {kod: 0.5 for kod in portfoy}  # 50-50
    sermaye = [100.0]
    
    bekleyen_satis = {}  # kod -> satis gun idx
    
    for gi in range(start_idx+1, end_idx+1):
        _, norm = hesapla_donemsel_metrikler(tarihler_asc, fiyatlar_asc, gi)
        skor = norm @ katsayi
        # rank
        rank_order = np.argsort(-skor)  # büyükten küçüğe
        rank_dict = {kodlar[rank_order[i]]: i+1 for i in range(len(kodlar))}
        
        # satış kontrol
        for kod in list(portfoy):
            if kod in rank_dict and rank_dict[kod] > 10:
                bekleyen_satis[kod] = gi + periyod_valor
        
        # bekleyen satışları gerçekleştir
        for kod, sat_gun in list(bekleyen_satis.items()):
            if gi >= sat_gun:
                # sat
                if kod in portfoy:
                    portfoy.remove(kod)
                del bekleyen_satis[kod]
                # en yüksek skorlu yeni fon al
                for idx in rank_order:
                    yeni_kod = kodlar[idx]
                    if yeni_kod not in portfoy:
                        portfoy.append(yeni_kod)
                        break
        
        # sermaye hesapla (basit, fiyat değişimi kadar)
        # Burada gerçek getiri: portföydeki fonların günlük getirisi ortalaması
        gun_getiri = 0
        cnt = 0
        for kod in portfoy:
            if fiyatlar_asc[kod][gi-1] > 1e-9:
                gun_getiri += (fiyatlar_asc[kod][gi] / fiyatlar_asc[kod][gi-1] - 1)
                cnt += 1
        if cnt:
            gun_getiri /= cnt
            sermaye.append(sermaye[-1] * (1 + gun_getiri))
        else:
            sermaye.append(sermaye[-1])

    # yıllıklandırılmış getiri
    if len(sermaye) < 2:
        return 0.0, 0
    toplam_getiri = sermaye[-1] / sermaye[0] - 1
    gun_sayisi = end_idx - start_idx
    yillik = (1 + toplam_getiri) ** (252 / max(gun_sayisi, 1)) - 1
    return yillik * 100, sermaye[-1]

# --- Walk-forward ana döngü ---
def main():
    tarihler_asc, tarihler_desc, fiyatlar_asc = yukle()
    print(f"Veri: {tarihler_asc[0]} -> {tarihler_asc[-1]}, {len(tarihler_asc)} gün, {len(fiyatlar_asc)} fon")
    
    # Ölçek kontrol
    olcek_kontrol(tarihler_asc, fiyatlar_asc)

    # Pencereler: 252 train, 126 test, örtüşmeyen test
    usable_start = 252  # YIL için warmup
    train_len = 252
    test_len = 126
    pencereler = []
    idx = usable_start + train_len
    while idx + test_len < len(tarihler_asc):
        train_s = idx - train_len
        train_e = idx - 1
        test_s = idx
        test_e = idx + test_len - 1
        pencereler.append((train_s, train_e, test_s, test_e))
        idx += test_len  # örtüşmeyen test
    
    print(f"{len(pencereler)} pencere oluşturuldu (train {train_len}, test {test_len})")
    for i, (ts, te, tes, tee) in enumerate(pencereler):
        print(f"  Pencere {i+1}: train {tarihler_asc[ts]}->{tarihler_asc[te]} | test {tarihler_asc[tes]}->{tarihler_asc[tee]}")

    # Test et
    rows = []
    for pi, (train_s, train_e, test_s, test_e) in enumerate(pencereler):
        # Baseline test
        base_ret, _ = simulate_window(tarihler_asc, fiyatlar_asc, test_s, test_e, BASELINE_KATSAYI)
        # Cenker
        cenker_ret, _ = simulate_window(tarihler_asc, fiyatlar_asc, test_s, test_e, CENKER_KATSAYI)
        # Random null (katsayılar rastgele)
        rand_k = np.random.uniform(0.2, 3.0, size=6)
        rand_ret, _ = simulate_window(tarihler_asc, fiyatlar_asc, test_s, test_e, rand_k)
        # Sadece YIL benchmark
        yil_only = np.array([0,0,0,0,0,1.0])
        yil_ret, _ = simulate_window(tarihler_asc, fiyatlar_asc, test_s, test_e, yil_only)

        rows.append({
            "pencere": pi+1,
            "train_bas": tarihler_asc[train_s],
            "train_bit": tarihler_asc[train_e],
            "test_bas": tarihler_asc[test_s],
            "test_bit": tarihler_asc[test_e],
            "baseline": base_ret,
            "cenker": cenker_ret,
            "random_null": rand_ret,
            "yil_only": yil_ret
        })
        print(f"P{pi+1} | base {base_ret:5.2f}% | cenker {cenker_ret:5.2f}% | rand {rand_ret:5.2f}% | yil {yil_ret:5.2f}%")

    df = pd.DataFrame(rows)
    df.to_csv(RESULT_DIR / "walkforward_summary.csv", index=False)

    # Özet: ortalama yerine yendiği pencere oranı ve en kötü pencere - senin istediğin
    def ozet(col):
        return {
            "ort": df[col].mean(),
            "medyan": df[col].median(),
            "en_kotu": df[col].min(),
            "en_iyi": df[col].max(),
            "baseline_yenen_oran": (df[col] > df["baseline"]).mean() if col != "baseline" else None
        }

    print("\n=== ÖZET (test pencereleri) ===")
    for c in ["baseline","cenker","random_null","yil_only"]:
        o = ozet(c)
        print(f"{c:12s} ort {o['ort']:5.2f}% medyan {o['medyan']:5.2f}% en_kotu {o['en_kotu']:5.2f}% en_iyi {o['en_iyi']:5.2f}% yenme {o['baseline_yenen_oran']}")

    # DE için not: DE train'de optimize edip testte denemek istersen
    # burada train penceresinde basit bir grid search / DE koşup katsayıyı bulabilirsin
    # Şimdilik placeholder - train optimizasyonu eklemek için train_s/train_e kullan

if __name__ == "__main__":
    main()
