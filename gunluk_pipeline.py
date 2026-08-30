"""
gunluk_pipeline.py
---------------------
Her gun otomatik calisacak ana betik. Sirasiyla:
1. TEFAS'tan bugunun fiyatlarini ceker
2. Fiyat gecmisini gunceller (yeni gunu ekler, en eskiyi atar -- 255 gunluk
   pencere korunur)
3. Donemsel getiri + Sharpe + MDD + Consistency + Otokorelasyon +
   EnBuyukGunOrani hesaplayip Z skorunu uretir
4. Tema kumelemeyi (TemaOlustur) calistirir
5. Sirala, dunku sirayi kaydet, docs/data.json'i yazar (dashboard'un
   okudugu dosya)

Bu betik Babur'un mudahalesine gerek kalmadan GitHub Actions tarafindan
her gun calistirilacak.
"""

import json
import datetime
from pathlib import Path

import numpy as np
from pytefas import Crawler, TefasAPIError, TefasRateLimitError

from puanlama_metrikleri import (
    donemsel_getiriler, sharpe, mdd, consistency, otokorelasyon,
    en_buyuk_gun_orani, puanlama_motoru, tema_olustur,
)

KOK = Path(__file__).parent
GUN_SAYISI = 255


def bugunun_tarihi_veya_son_is_gunu() -> str:
    bugun = datetime.date.today()
    while bugun.weekday() >= 5:
        bugun = bugun - datetime.timedelta(days=1)
    return bugun.strftime("%Y-%m-%d")


def yukle(dosya):
    with open(KOK / dosya, encoding="utf-8") as f:
        return json.load(f)


def kaydet(veri, dosya):
    with open(KOK / dosya, "w", encoding="utf-8") as f:
        json.dump(veri, f, ensure_ascii=False, indent=1)


def bugunku_fiyatlari_cek(fon_kodlari: set, tarih: str):
    """TEFAS'tan o gune ait tum fon fiyatlarini ceker, sadece takip
    listemizdeki fonlari dondurur: (fiyatlar_dict, tefas_tarihi).
    tefas_tarihi, TEFAS'in kendi dondurdugu 'date' alanindan okunur --
    yerel tarih tahminine guvenilmez (hafta sonu/tatil/henuz yayinlanmamis
    veri durumlarinda yanlis olabilir)."""
    tefas = Crawler()
    df = tefas.fetch(tarih, columns="info", kind="YAT")
    kayitlar = json.loads(df.to_json(orient="records"))
    sonuc = {}
    tefas_tarihi = None
    for k in kayitlar:
        kod = k.get("fund_code")
        if kod in fon_kodlari:
            sonuc[kod] = k.get("price")
        if tefas_tarihi is None and k.get("date"):
            tefas_tarihi = str(k["date"])[:10]
    return sonuc, tefas_tarihi


def main():
    fon_listesi = yukle("fon_listesi.json")
    sabitler = yukle("sabitler.json")
    fiyat_gecmisi = yukle("price_history.json")
    try:
        onceki_rank = yukle("onceki_rank.json")
    except FileNotFoundError:
        onceki_rank = {}
    try:
        portfoy = yukle("portfoy.json")  # {kod: adet} -- elle guncellenen elde tutulan fonlar
    except FileNotFoundError:
        portfoy = {}

    tarih_tahmin = bugunun_tarihi_veya_son_is_gunu()
    print(f"Tahmini tarih: {tarih_tahmin} (TEFAS'in kendi tarihiyle dogrulanacak)")

    print("TEFAS'tan bugunun fiyatlari cekiliyor...")
    try:
        bugun_fiyat, tefas_tarihi = bugunku_fiyatlari_cek(set(fon_listesi.keys()), tarih_tahmin)
    except (TefasRateLimitError, TefasAPIError) as e:
        print(f"HATA: TEFAS'tan veri cekilemedi: {e}")
        return

    if not tefas_tarihi:
        print("HATA: TEFAS'tan tarih bilgisi alinamadi, islem durduruldu.")
        return

    son_bilinen_tarih = fiyat_gecmisi["tarihler"][0] if fiyat_gecmisi["tarihler"] else None
    yeni_gun_var_mi = (tefas_tarihi != son_bilinen_tarih)
    tarih = tefas_tarihi
    print(f"TEFAS'in gercek veri tarihi: {tarih} (elimizdeki en yeni: {son_bilinen_tarih}) "
          f"-> {'YENI GUN' if yeni_gun_var_mi else 'henuz yeni veri yok, pencere kaydirilmayacak'}")

    eksik = [k for k in fon_listesi if k not in bugun_fiyat]
    if eksik:
        print(f"UYARI: {len(eksik)} fon icin bugun fiyat gelmedi (islem "
              f"gormemis olabilir): {eksik[:10]}{'...' if len(eksik)>10 else ''}")

    # --- fiyat gecmisini SADECE gercekten yeni bir gun varsa guncelle ---
    if yeni_gun_var_mi:
        fiyat_gecmisi["tarihler"].insert(0, tarih)
        fiyat_gecmisi["tarihler"] = fiyat_gecmisi["tarihler"][:GUN_SAYISI]
        for kod in fon_listesi:
            eski_seri = fiyat_gecmisi["fiyatlar"].get(kod, [0.0] * GUN_SAYISI)
            yeni_fiyat = bugun_fiyat.get(kod, eski_seri[0] if eski_seri else 0.0)
            eski_seri = [yeni_fiyat] + eski_seri
            fiyat_gecmisi["fiyatlar"][kod] = eski_seri[:GUN_SAYISI]
    else:
        print("Pencere kaydirilmadi -- ayni gun icin sadece skorlar yeniden hesaplanacak.")

    kodlar = list(fon_listesi.keys())
    fiyat_full = np.array([fiyat_gecmisi["fiyatlar"][k] for k in kodlar])
    n = sabitler["periyod"]
    fiyat_np1 = fiyat_full[:, : n + 1]
    fiyat_n = fiyat_full[:, :n]

    print("Skorlar hesaplaniyor...")
    k_sigmoid = sabitler["k"]
    dk = sabitler["donemsel_katsayilar"]
    Z = np.zeros(len(kodlar))
    donemsel = donemsel_getiriler(fiyat_full)
    donem_map = {"gunluk": "gunluk", "haftalik": "haftalik", "aylik": "aylik",
                 "uc_aylik": "uc_aylik", "alti_aylik": "alti_aylik", "yillik": "yillik"}
    for ad, kat_anahtar in donem_map.items():
        Z += puanlama_motoru(donemsel[ad], yon=1, katsayi=dk[kat_anahtar], k=k_sigmoid, winsor=True)

    topk6 = sum(dk.values())
    oran = sabitler["risk_metrik_oranlari"]
    Z += puanlama_motoru(sharpe(fiyat_np1, sabitler["risksiz_yillik"]), yon=1,
                          katsayi=topk6 * oran["sharpe"], k=k_sigmoid, winsor=True)
    Z += puanlama_motoru(mdd(fiyat_n), yon=-1,
                          katsayi=topk6 * oran["mdd"], k=k_sigmoid, winsor=False)
    Z += puanlama_motoru(consistency(fiyat_np1), yon=1,
                          katsayi=topk6 * oran["consistency"], k=k_sigmoid, winsor=True)
    Z += puanlama_motoru(otokorelasyon(fiyat_np1), yon=-1,
                          katsayi=topk6 * oran["otokorelasyon"], k=k_sigmoid, winsor=True)
    Z += puanlama_motoru(en_buyuk_gun_orani(fiyat_np1), yon=-1,
                          katsayi=topk6 * oran["ebg"], k=k_sigmoid, winsor=False)

    print("Tema kumeleme calisiyor...")
    valorler = np.array([fon_listesi[k].get("valor") for k in kodlar], dtype=object)
    fiyat_91 = fiyat_full[:, :91]
    tema_sonuc = dict(tema_olustur(fiyat_91, kodlar, valorler))

    # --- Is gunu sayilari (TarihKontrol'un G216/G217 karsiligi) ---
    # tarih formati YYYY-MM-DD; ay/yil karsilastirmasi bugunku (en yeni) tarihe gore
    bugun_ay, bugun_yil = tarih[:7], tarih[:4]
    ay_sayisi = sum(1 for t in fiyat_gecmisi["tarihler"] if t and t[:7] == bugun_ay)
    yil_sayisi = sum(1 for t in fiyat_gecmisi["tarihler"] if t and t[:4] == bugun_yil)

    sira = np.argsort(-Z)
    yeni_rank = {}
    dashboard_funds = []
    for sirano, idx in enumerate(sira, start=1):
        kod = kodlar[idx]
        yeni_rank[kod] = sirano
        dunku = onceki_rank.get(kod)
        adet = portfoy.get(kod, 0)
        fiyat_bugun = float(fiyat_full[idx, 0])
        fiyat_dun = float(fiyat_full[idx, 1]) if fiyat_full.shape[1] > 1 else fiyat_bugun
        guncel = round(adet * fiyat_bugun, 2)
        gunluk_tl = round(adet * (fiyat_bugun - fiyat_dun), 2)
        dashboard_funds.append({
            "kod": kod,
            "ad": fon_listesi[kod].get("ad"),
            "val": fon_listesi[kod].get("valor"),
            "rnk": sirano,
            "dunRnk": dunku,
            "isBold": bool(dunku is not None and sirano <= dunku),
            "gun": round(float(donemsel["gunluk"][idx]), 4),
            "haf": round(float(donemsel["haftalik"][idx]), 4),
            "ay": round(float(donemsel["aylik"][idx]), 4),
            "ay3": round(float(donemsel["uc_aylik"][idx]), 4),
            "ay6": round(float(donemsel["alti_aylik"][idx]), 4),
            "yil": round(float(donemsel["yillik"][idx]), 4),
            "tema": tema_sonuc.get(kod),
            "guncel": guncel,       # elde tutulan fonun bugunku TL degeri (adet x fiyat)
            "toplam": guncel,       # simdilik guncel ile ayni (birikmis kar/zarar takibi sonraki asama)
            "gunlukTL": gunluk_tl,  # bugunku TL bazli kar/zarar (adet x fiyat degisimi)
        })

    portfolio = [f for f in dashboard_funds if f["guncel"] and f["guncel"] > 0]
    totals = {
        "daily": round(sum(f["gunlukTL"] for f in portfolio), 2),
        "monthly": round(sum(f["guncel"] * f["ay"] / 100 for f in portfolio), 2),
        "yearly": round(sum(f["guncel"] * f["yil"] / 100 for f in portfolio), 2),
    }

    data_json = {
        "funds": dashboard_funds,
        "portfolio": portfolio,
        "totals": totals,
        "timestamp": int(datetime.datetime.now().timestamp() * 1000),
        "tableDate": tarih,
        "isGunuSayilari": {"aylik": ay_sayisi, "yillik": yil_sayisi},
    }

    kaydet(fiyat_gecmisi, "price_history.json")
    kaydet(yeni_rank, "onceki_rank.json")
    kaydet(data_json, "docs/data.json")
    print(f"Tamamlandi: {len(dashboard_funds)} fon icin docs/data.json yazildi.")


if __name__ == "__main__":
    main()
