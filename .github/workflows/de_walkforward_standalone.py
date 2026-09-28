"""
de_walkforward_standalone.py
------------------------------
Bagimsiz, kesintisiz calisacak DE (Differential Evolution) gunluk
walk-forward dogrulama script'i. Claude sohbetinde (300sn/komut limiti
yuzunden) parca parca calistirdigimiz testin, kendi makinende tek
seferde, muhtemelen tum gece boyunca calisacak tam halidir.

KESILIRSE KALDIGI YERDEN DEVAM EDER: her gunun sonucu ciktikca
CHECKPOINT_FILE'a yaziliyor; script yeniden calistirilinca zaten
tamamlanmis gunleri atlar.

Gereken dosyalar (bu script ile ayni klasorde olmali):
  - price_history.json
  - puanlama_metrikleri.py

Kullanim:
  python3 de_walkforward_standalone.py

Ayarlari asagidaki "AYARLAR" bolumunden degistirebilirsin.
"""
import json
import pickle
import time
import os
import sys
import numpy as np
from scipy.optimize import differential_evolution
from puanlama_metrikleri import puanlama_motoru, donemsel_getiriler, sortino

# ============================== AYARLAR =====================================
PRICE_HISTORY_JSON = "price_history.json"
CHECKPOINT_FILE = "de_wf_full_checkpoint.pkl"
LOG_FILE = "de_wf_full_log.txt"

EXCLUDE_FONLAR = {"BBR", "END", "BRG", "YVD"}   # portfoy/endeks/PPF kodlari

RISKSIZ_YILLIK = 0.42
SORTINO_PENCERE_BASLANGIC = 69     # baslangic degeri, katsayi aramasina dahil degil (istersen DE'ye ekleyebilirsin)
K_SIGMOID = 3

TRAIN_GUN = 189          # ~9 ay, her gun icin geriye donuk kayan train penceresi
TEST_GUN = 252           # ~1 yil -- SORULAN SENARYO: son 1 yili gunluk walk-forward test et
                          # (6 ay icin 126, 3 ay icin 63 yapabilirsin)

PORTFOY_BOYUTU = 2

# DE ayarlari -- Cenker'in gercek kurulumuna kiyasla hala hafif (165 degil, ~35-50
# populasyon; 800 fon degil, gercek fon evrenin; cok yillik degil, sadece TRAIN_GUN)
DE_POPSIZE = 5           # populasyon = DE_POPSIZE * 7 parametre
DE_MAXITER = 8
DE_SEED = 42

BASELINE_KATSAYILAR = [0.43, 0.67, 2.41, 2.25, 1.55, 2.53, 1.5]  # Gun/Haf/Ay/3Ay/6Ay/Yil/Sortino
METRIKLER = ["gunluk", "haftalik", "aylik", "uc_aylik", "alti_aylik", "yillik", "sortino"]
DE_BOUNDS = [(0.05, 6.0)] * 7
# =============================================================================


def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def load_universe():
    """
    Fon gecmisleri farkli uzunlukta olabilir (yeni listelenen fonlar daha kisa).
    Once TRAIN_GUN+TEST_GUN+LOOKBACK icin gereken minimum uzunlugu (`ihtiyac`)
    karsilayan fonlari sec, sonra HEPSINI bu grubun en kisasina (>= ihtiyac)
    kirp -- boylece tek bir yeni fon yuzunden tum analiz penceresi kucul mez.
    """
    LOOKBACK_ICIN = 253
    ihtiyac = LOOKBACK_ICIN + TRAIN_GUN + TEST_GUN

    with open(PRICE_HISTORY_JSON, encoding="utf-8") as f:
        data = json.load(f)
    fiyatlar = data["fiyatlar"]

    tum_arrs = {}
    for k, v in fiyatlar.items():
        if k in EXCLUDE_FONLAR:
            continue
        arr = np.array(v, dtype=float)
        if (arr <= 0).any():
            continue
        tum_arrs[k] = arr

    yeterli = {k: a for k, a in tum_arrs.items() if len(a) >= ihtiyac}

    if not yeterli:
        # Istenen TEST_GUN+TRAIN_GUN icin yeterli derinlikte fon yok --
        # elimizdeki en iyi ortak uzunluga geriliyoruz (TEST_GUN otomatik kisalacak).
        min_kabul = LOOKBACK_ICIN + TRAIN_GUN + 1  # en az 1 test gunu
        yeterli = {k: a for k, a in tum_arrs.items() if len(a) >= min_kabul}
        if not yeterli:
            raise RuntimeError(
                f"Hicbir fonda LOOKBACK+TRAIN icin yeterli gecmis yok "
                f"(gereken >= {min_kabul} gun). price_history.json'u kontrol et."
            )
        print(f"UYARI: istenen TEST_GUN={TEST_GUN} icin yeterli derinlikte fon yok, "
              f"daha kisa bir ortak uzunluga geriliyoruz.", flush=True)

    ortak_len = min(len(a) for a in yeterli.values())
    kodlar = list(yeterli.keys())
    rows = [yeterli[k][:ortak_len] for k in kodlar]

    print(f"Bilgi: {len(kodlar)} fon secildi (digerleri yetersiz gecmis nedeniyle "
          f"disarida birakildi), ortak uzunluk = {ortak_len} gun.", flush=True)

    P_newest_first = np.array(rows)   # (n_fon, n_gun), index0 = en yeni
    C = P_newest_first[:, ::-1].copy()  # kronolojik: t=0 en eski
    return kodlar, C


def precompute(C, sim_days):
    """Her gun icin ham periyodik getiriler + sortino + ileri (t->t+1) getiri."""
    raw, fwd_ret = {}, {}
    n_gun = C.shape[1]
    for i, t in enumerate(sim_days):
        hist = C[:, : t + 1][:, ::-1]
        pencereler = donemsel_getiriler(hist)
        sort_pencere = min(SORTINO_PENCERE_BASLANGIC, hist.shape[1] - 1)
        sort_val = sortino(hist[:, : sort_pencere + 1], RISKSIZ_YILLIK, sort_pencere)
        raw[t] = {**{m: pencereler[m] for m in ["gunluk", "haftalik", "aylik",
                                                   "uc_aylik", "alti_aylik", "yillik"]},
                  "sortino": sort_val}
        fwd_ret[t] = (C[:, t + 1] - C[:, t]) / C[:, t]
        if (i + 1) % 50 == 0:
            log(f"  [on-hesap] {i+1}/{len(sim_days)} gun islendi")
    return raw, fwd_ret


def score_day(raw, t, coeffs):
    r = raw[t]
    toplam = np.zeros_like(r["gunluk"])
    for i, m in enumerate(METRIKLER):
        toplam = toplam + puanlama_motoru(r[m], yon=1, katsayi=coeffs[i], k=K_SIGMOID, winsor=True)
    return toplam


def simulate(raw, fwd_ret, days, coeffs, portfoy_boyutu=PORTFOY_BOYUTU):
    sermaye = 1.0
    for t in days:
        skor = score_day(raw, t, coeffs)
        top_idx = np.argsort(-skor)[:portfoy_boyutu]
        sermaye *= (1 + fwd_ret[t][top_idx].mean())
    return (sermaye - 1) * 100


def gunluk_getiri_tek(raw, fwd_ret, t, coeffs, portfoy_boyutu=PORTFOY_BOYUTU):
    skor = score_day(raw, t, coeffs)
    top_idx = np.argsort(-skor)[:portfoy_boyutu]
    return fwd_ret[t][top_idx].mean()


def main():
    t_start = time.time()
    log(f"\n=== Calisma basladi: {time.strftime('%Y-%m-%d %H:%M:%S')} ===")

    kodlar, C = load_universe()
    n_fon, n_gun = C.shape
    log(f"Fon evreni: {n_fon} fon, {n_gun} gun")

    LOOKBACK = 253
    t_min = LOOKBACK - 1
    t_max = n_gun - 2
    all_sim_days = list(range(t_min, t_max + 1))

    if len(all_sim_days) < TRAIN_GUN + TEST_GUN:
        log(f"UYARI: istenen TRAIN_GUN+TEST_GUN ({TRAIN_GUN+TEST_GUN}) mevcut veriden "
            f"({len(all_sim_days)}) fazla. TEST_GUN otomatik kisaltiliyor.")
        test_gun_eff = max(1, len(all_sim_days) - TRAIN_GUN)
    else:
        test_gun_eff = TEST_GUN

    test_days_full = all_sim_days[-test_gun_eff:]
    precompute_gerekli_gunler = list(range(all_sim_days[-1] - TRAIN_GUN - test_gun_eff + 1, all_sim_days[-1] + 1))
    precompute_gerekli_gunler = [t for t in precompute_gerekli_gunler if t in all_sim_days or t >= t_min]

    # guvenli: precompute'i ihtiyac olan tum t araligi icin yap
    t_ihtiyac_min = test_days_full[0] - TRAIN_GUN
    t_ihtiyac_min = max(t_ihtiyac_min, t_min)
    precompute_days = list(range(t_ihtiyac_min, all_sim_days[-1] + 1))

    log(f"Test gun sayisi: {len(test_days_full)}  |  On-hesap gun sayisi: {len(precompute_days)}")
    log("Ham metrikler hesaplaniyor (bir kereye mahsus)...")
    raw, fwd_ret = precompute(C, precompute_days)
    log("On-hesap tamamlandi.")

    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "rb") as f:
            state = pickle.load(f)
        log(f"Checkpoint bulundu: {len(state['gunluk_log'])} gun zaten tamamlanmis, devam ediliyor.")
    else:
        state = {"sermaye_wf": 1.0, "gunluk_log": []}

    tamamlanan_t = {row[0] for row in state["gunluk_log"]}

    for j, test_t in enumerate(test_days_full):
        if test_t in tamamlanan_t:
            continue

        train_window = list(range(test_t - TRAIN_GUN, test_t))

        def neg_obj(c):
            return -simulate(raw, fwd_ret, train_window, c)

        gun_t0 = time.time()
        result = differential_evolution(
            neg_obj, DE_BOUNDS, maxiter=DE_MAXITER, popsize=DE_POPSIZE,
            tol=1e-3, seed=DE_SEED, polish=False, updating="deferred", workers=1,
        )
        coeffs = list(result.x)
        g = gunluk_getiri_tek(raw, fwd_ret, test_t, coeffs)
        state["sermaye_wf"] *= (1 + g)
        state["gunluk_log"].append((test_t, g, coeffs))

        with open(CHECKPOINT_FILE, "wb") as f:
            pickle.dump(state, f)

        log(f"gun {j+1}/{len(test_days_full)} (t={test_t}): {g*100:+.3f}%  "
            f"kumulatif={((state['sermaye_wf']-1)*100):+.2f}%  "
            f"sure={time.time()-gun_t0:.1f}sn  katsayilar={[round(c,2) for c in coeffs]}")

    toplam_getiri = (state["sermaye_wf"] - 1) * 100
    baseline_getiri = simulate(raw, fwd_ret, test_days_full, BASELINE_KATSAYILAR)

    log("\n=== SONUC ===")
    log(f"Test donemi: {len(test_days_full)} is gunu")
    log(f"DE gunluk walk-forward kumulatif getiri : {toplam_getiri:+.2f}%")
    log(f"Baseline (sabit, optimize edilmemis)     : {baseline_getiri:+.2f}%")
    log(f"Toplam sure: {(time.time()-t_start)/60:.1f} dakika")


if __name__ == "__main__":
    main()
