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


def yeni_fon_gecmisini_cek(kod: str, gun_sayisi: int = GUN_SAYISI):
    """Yeni eklenen bir fon icin gecmis fiyat serisini TEFAS'tan ceker.
    Donus: (fon_adi, fiyatlar_listesi_yeniden_eskiye) ya da (None, None)."""
    bugun = datetime.date.today()
    baslangic = bugun - datetime.timedelta(days=int(gun_sayisi * 1.5))  # hafta sonlari icin pay
    tefas = Crawler()
    df = tefas.fetch(baslangic.strftime("%Y-%m-%d"), bugun.strftime("%Y-%m-%d"),
                      columns="info", kind="YAT", fund_code=kod)
    kayitlar = json.loads(df.to_json(orient="records"))
    if not kayitlar:
        return None, None
    # tarihe gore yeniden eskiye sirala
    kayitlar.sort(key=lambda k: k.get("date", ""), reverse=True)
    fiyatlar = [k.get("price", 0.0) for k in kayitlar][:gun_sayisi]
    ad = kayitlar[0].get("fund_name")
    if len(fiyatlar) < gun_sayisi:
        # yetersiz gecmis (yeni kurulmus fon olabilir) -- ilk degerle doldur
        doldurma = fiyatlar[-1] if fiyatlar else 0.0
        fiyatlar = fiyatlar + [doldurma] * (gun_sayisi - len(fiyatlar))
    return ad, fiyatlar


def yukle(dosya):
    with open(KOK / dosya, encoding="utf-8") as f:
        return json.load(f)


def tarihte_fiyat(tarihler, fiyatlar, hedef_tarih):
    """tarihler (en yeniden eskiye siralı) icinde hedef_tarih'e esit ya da
    ondan once gelen ilk (en yakin) fiyati dondurur. Bulunamazsa None."""
    for i, t in enumerate(tarihler):
        if t and t <= hedef_tarih:
            return fiyatlar[i]
    return None


def ay_yil_baslangici(tarih_str):
    """'YYYY-MM-DD' -> (ay_baslangici, yil_baslangici)ayni formatta."""
    yil, ay = tarih_str[:4], tarih_str[5:7]
    return f"{yil}-{ay}-01", f"{yil}-01-01"


def is_gunu_ekle(tarih_str, gun_sayisi):
    """tarih_str ('YYYY-MM-DD') uzerine gun_sayisi kadar IS GUNU (hafta
    sonu haric) ekler. Valor hesaplamasi icin (resmi tatiller dahil
    edilmiyor, yaklasik bir tahmindir)."""
    tarih = datetime.date.fromisoformat(tarih_str)
    kalan = int(gun_sayisi)
    while kalan > 0:
        tarih += datetime.timedelta(days=1)
        if tarih.weekday() < 5:  # 0-4 = Pazartesi-Cuma
            kalan -= 1
    return tarih.strftime("%Y-%m-%d")


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
        portfoy = yukle("portfoy.json")  # {kod: {adet, alis_tarihi?}} -- elde tutulan fonlar
    except FileNotFoundError:
        portfoy = {}
    try:
        bekleyen_emirler = yukle("bekleyen_emirler.json")
    except FileNotFoundError:
        bekleyen_emirler = []
    try:
        bekleyen_valorler = yukle("bekleyen_valorler.json")
    except FileNotFoundError:
        bekleyen_valorler = []
    try:
        eklenecek_fonlar = yukle("eklenecek_fonlar.json")
    except FileNotFoundError:
        eklenecek_fonlar = []

    # --- yeni eklenen fonlarin 1 yillik gecmisini cek ---
    if eklenecek_fonlar:
        print(f"{len(eklenecek_fonlar)} yeni fon icin gecmis veri cekiliyor...")
        kalan_eklenecek = []
        for istek in eklenecek_fonlar:
            kod = istek.get("kod")
            valor = istek.get("valor")
            if kod in fon_listesi:
                print(f"  {kod} zaten listede, atlandi.")
                continue
            try:
                ad, fiyatlar = yeni_fon_gecmisini_cek(kod)
            except (TefasRateLimitError, TefasAPIError) as e:
                print(f"  HATA: {kod} icin veri cekilemedi ({e}), tekrar denenecek.")
                kalan_eklenecek.append(istek)
                continue
            if not ad:
                print(f"  UYARI: {kod} icin TEFAS'ta veri bulunamadi, kod hatali olabilir.")
                continue
            fon_listesi[kod] = {"ad": ad, "valor": valor}
            fiyat_gecmisi["fiyatlar"][kod] = fiyatlar
            print(f"  {kod} ({ad}) eklendi, {len(fiyatlar)} gunluk gecmisle.")
        eklenecek_fonlar = kalan_eklenecek
        kaydet(eklenecek_fonlar, "eklenecek_fonlar.json")
        kaydet(fon_listesi, "fon_listesi.json")
        kaydet(fiyat_gecmisi, "price_history.json")

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

    eksik = [k for k in fon_listesi if k not in bugun_fiyat and k not in ("END", "BBR")]
    if eksik:
        print(f"UYARI: {len(eksik)} fon icin bugun fiyat gelmedi (islem "
              f"gormemis olabilir): {eksik[:10]}{'...' if len(eksik)>10 else ''}")

    # --- END sentetik endeksi (Endeks.bas / EndeksHesapla ile ayni mantik) ---
    # END'in "fiyati" gercek bir TEFAS fiyati degil; tum evrenin (END, BBR
    # haric) esit-agirlikli ortalama gunluk getirisiyle zincirlenerek
    # hesaplanan sentetik bir gosterge. Sadece gercekten yeni bir gun
    # varsa hesaplanir (ayni gun icin tekrar tekrar zincirlenmesin diye).
    if yeni_gun_var_mi:
        haric = {"END", "BBR"}
        getiriler = []
        for kod in fon_listesi:
            if kod in haric:
                continue
            dun_fiyat = fiyat_gecmisi["fiyatlar"].get(kod, [0.0])[0]
            bugun_fiyat_kod = bugun_fiyat.get(kod)
            if dun_fiyat and bugun_fiyat_kod:
                getiriler.append((bugun_fiyat_kod - dun_fiyat) / dun_fiyat * 100)
        if getiriler:
            ort_getiri = sum(getiriler) / len(getiriler)
            end_dun = fiyat_gecmisi["fiyatlar"].get("END", [0.0])[0]
            if not end_dun or end_dun <= 0:
                end_dun = 100.0  # ilk calistirma tabani
            bugun_fiyat["END"] = end_dun * (1 + ort_getiri / 100)
            print(f"END guncellendi: ortalama getiri %{ort_getiri:.4f}, "
                  f"yeni deger {bugun_fiyat['END']:.4f}")

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

    # --- bekleyen AL/SAT emirlerini isle (Rutin'in 212/213 mantiginin karsiligi) ---
    if bekleyen_emirler:
        print(f"{len(bekleyen_emirler)} bekleyen emir isleniyor...")
        kalan_emirler = []
        for emir in bekleyen_emirler:
            kod = emir.get("kod")
            tip = emir.get("tip")
            fiyat_bugun_kod = bugun_fiyat.get(kod)
            if fiyat_bugun_kod is None or fiyat_bugun_kod <= 0:
                print(f"  UYARI: {kod} icin bugun fiyat yok, emir ertelendi.")
                kalan_emirler.append(emir)
                continue
            if tip == "AL":
                tutar = emir.get("tutar", 0)
                adet = int(tutar / fiyat_bugun_kod)
                portfoy[kod] = {"adet": adet, "alis_tarihi": tarih}
                print(f"  AL: {kod} {adet} adet ({tutar} TL / {fiyat_bugun_kod})")
            elif tip == "SAT":
                satilacak_adet = emir.get("adet", 0)
                mevcut = portfoy.get(kod)
                mevcut_adet = mevcut.get("adet", 0) if isinstance(mevcut, dict) else (mevcut or 0)
                kalan_adet = mevcut_adet - satilacak_adet
                satis_tutari = round(satilacak_adet * fiyat_bugun_kod, 2)
                valor_gun = fon_listesi.get(kod, {}).get("valor") or 0
                hesaba_gecis = is_gunu_ekle(tarih, valor_gun)
                bekleyen_valorler.append({
                    "kod": kod, "tutar": satis_tutari,
                    "satis_tarihi": tarih, "hesaba_gecis_tarihi": hesaba_gecis,
                })
                if kalan_adet <= 0:
                    if kod in portfoy:
                        del portfoy[kod]
                    print(f"  SAT: {kod} pozisyonu tamamen kapatildi "
                          f"({satilacak_adet} adet satildi, {mevcut_adet} adet vardi, "
                          f"{satis_tutari} TL {hesaba_gecis} tarihinde hesaba gecer)")
                else:
                    alis_tarihi_eski = mevcut.get("alis_tarihi") if isinstance(mevcut, dict) else None
                    portfoy[kod] = {"adet": kalan_adet, "alis_tarihi": alis_tarihi_eski}
                    print(f"  SAT: {kod} kismi satis, {satilacak_adet} adet satildi, "
                          f"{kalan_adet} adet kaldi (alis tarihi korunuyor), "
                          f"{satis_tutari} TL {hesaba_gecis} tarihinde hesaba gecer)")
        bekleyen_emirler = kalan_emirler  # sadece ertelenenler kalir
        kaydet(bekleyen_emirler, "bekleyen_emirler.json")
        kaydet(portfoy, "portfoy.json")

    # --- suresi gecmis (hesaba gecmis) valor hatirlaticilarini temizle ---
    bekleyen_valorler = [v for v in bekleyen_valorler if v["hesaba_gecis_tarihi"] > tarih]
    kaydet(bekleyen_valorler, "bekleyen_valorler.json")

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
    ay_baslangic, yil_baslangic = ay_yil_baslangici(tarih)
    for sirano, idx in enumerate(sira, start=1):
        kod = kodlar[idx]
        yeni_rank[kod] = sirano
        dunku = onceki_rank.get(kod)
        pozisyon = portfoy.get(kod)
        adet = pozisyon.get("adet", 0) if isinstance(pozisyon, dict) else (pozisyon or 0)
        alis_tarihi = pozisyon.get("alis_tarihi") if isinstance(pozisyon, dict) else None
        fiyat_bugun = float(fiyat_full[idx, 0])
        fiyat_dun = float(fiyat_full[idx, 1]) if fiyat_full.shape[1] > 1 else fiyat_bugun
        guncel = round(adet * fiyat_bugun, 2)
        gunluk_tl = round(adet * (fiyat_bugun - fiyat_dun), 2)

        # Maliyet bazli kar/zarar: SADECE tek seferde (blok) alinan fonlar icin
        # (alis_tarihi biliniyorsa). PPF gibi kismi alim-satim yapilan fonlarda
        # (alis_tarihi yok) tek bir maliyet fiyati olmadigindan hesaplanamaz --
        # yanlis sayi gostermektense hic gostermiyoruz (None).
        ay_kar = yil_kar = None
        if adet > 0 and alis_tarihi:
            ay_baz = max(alis_tarihi, ay_baslangic)
            yil_baz = max(alis_tarihi, yil_baslangic)
            fiyat_ay_baz = tarihte_fiyat(fiyat_gecmisi["tarihler"], fiyat_gecmisi["fiyatlar"][kod], ay_baz)
            fiyat_yil_baz = tarihte_fiyat(fiyat_gecmisi["tarihler"], fiyat_gecmisi["fiyatlar"][kod], yil_baz)
            if fiyat_ay_baz:
                ay_kar = round(adet * (fiyat_bugun - fiyat_ay_baz), 2)
            if fiyat_yil_baz:
                yil_kar = round(adet * (fiyat_bugun - fiyat_yil_baz), 2)

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
            "guncel": guncel,        # elde tutulan fonun bugunku TL degeri (adet x fiyat)
            "toplam": guncel,
            "gunlukTL": gunluk_tl,   # bugunku TL bazli kar/zarar (maliyete ihtiyac duymaz)
            "ayKarZarar": ay_kar,    # None ise: PPF/kismi alim -- hesaplanamiyor
            "yilKarZarar": yil_kar,
        })


    portfolio = [f for f in dashboard_funds if f["guncel"] and f["guncel"] > 0]
    totals = {
        "daily": round(sum(f["gunlukTL"] for f in portfolio), 2),
        "monthly": round(sum(f["ayKarZarar"] for f in portfolio if f["ayKarZarar"] is not None), 2),
        "yearly": round(sum(f["yilKarZarar"] for f in portfolio if f["yilKarZarar"] is not None), 2),
    }

    data_json = {
        "funds": dashboard_funds,
        "portfolio": portfolio,
        "totals": totals,
        "timestamp": int(datetime.datetime.now().timestamp() * 1000),
        "tableDate": tarih,
        "isGunuSayilari": {"aylik": ay_sayisi, "yillik": yil_sayisi},
        "bekleyenValorler": bekleyen_valorler,
    }

    kaydet(fiyat_gecmisi, "price_history.json")
    kaydet(yeni_rank, "onceki_rank.json")
    kaydet(data_json, "docs/data.json")
    print(f"Tamamlandi: {len(dashboard_funds)} fon icin docs/data.json yazildi.")


if __name__ == "__main__":
    main()
