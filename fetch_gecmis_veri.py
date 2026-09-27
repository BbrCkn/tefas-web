"""
fetch_gecmis_veri.py -- TEK SEFERLIK script.
price_history.json'daki en eski tarihten GERIYE dogru, her fon icin
ayri ayri (yeni_fon_gecmisini_cek ile ayni yontem: Crawler().fetch ile
genis tarih araligi + fund_code) veri ceker ve price_history.json'un
SONUNA (daha eski tarihler) ekler.

Kullanim: GitHub Actions'ta workflow_dispatch ile bir kez calistirilir.
Repo KOKUNE konur (gunluk_pipeline.py ile ayni klasor).

Guvenlik: mevcut price_history.json'i BOZMAZ -- sadece daha eski
tarihleri sonuna ekler, var olan tarih/fiyatlara dokunmaz. Yine de
calistirmadan once repo'nun (git ile) bir onceki hali geri
alinabilir durumda olmali (GitHub zaten commit gecmisi tutuyor).
"""

import json
import time
import datetime
from pathlib import Path

from pytefas import Crawler, TefasAPIError, TefasRateLimitError

KOK = Path(__file__).parent

# Kac gun geriye gidilecek (takvim gunu, hafta sonlari dahil -- TEFAS
# zaten hafta sonu veri dondurmuyor, guvenli pay icin genis tutuldu)
HEDEF_TAKVIM_GUNU = 400

BEKLEME_SANIYE = 1.5          # her fon cagrisi arasi (rate-limit onlemi)
MAX_DENEME = 3                 # TefasRateLimitError'da tekrar deneme sayisi


def yukle(dosya):
    with open(KOK / dosya, encoding="utf-8") as f:
        return json.load(f)


def kaydet(veri, dosya):
    with open(KOK / dosya, "w", encoding="utf-8") as f:
        json.dump(veri, f, ensure_ascii=False, indent=1)


def _ts():
    return datetime.datetime.now().strftime("%H:%M:%S")


def fon_gecmisini_cek(kod: str, baslangic: str, bitis: str):
    """Tek fon icin [baslangic, bitis] araliginda (dahil) TEFAS'tan fiyat
    ceker. Donus: {tarih(YYYY-MM-DD): fiyat} sozlugu (bos olabilir)."""
    tefas = Crawler()
    for deneme in range(1, MAX_DENEME + 1):
        try:
            df = tefas.fetch(baslangic, bitis, columns="info", kind="YAT", fund_code=kod)
            break
        except TefasRateLimitError:
            bekleme = 10 * deneme
            print(f"  [{kod}] Rate limit, {bekleme}sn bekleniyor (deneme {deneme}/{MAX_DENEME})...")
            time.sleep(bekleme)
    else:
        print(f"  [{kod}] UYARI: {MAX_DENEME} denemede de basarisiz, atlandi.")
        return {}

    kayitlar = json.loads(df.to_json(orient="records"))
    sonuc = {}
    for k in kayitlar:
        tarih = str(k.get("date", ""))[:10]
        fiyat = k.get("price")
        if tarih and fiyat is not None:
            sonuc[tarih] = float(fiyat)
    return sonuc


def main():
    fon_listesi = yukle("fon_listesi.json")
    fiyat_gecmisi = yukle("price_history.json")

    en_eski_tarih = fiyat_gecmisi["tarihler"][-1]
    hedef_bitis = (datetime.date.fromisoformat(en_eski_tarih)
                   - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    hedef_baslangic = (datetime.date.fromisoformat(en_eski_tarih)
                       - datetime.timedelta(days=HEDEF_TAKVIM_GUNU)).strftime("%Y-%m-%d")

    print(f"[{_ts()}] Mevcut en eski tarih: {en_eski_tarih}")
    print(f"[{_ts()}] Hedef aralik: {hedef_baslangic} -> {hedef_bitis} "
          f"({len(fon_listesi)} fon icin tek tek cekilecek)")

    tum_fon_verisi = {}   # {kod: {tarih: fiyat}}
    for i, kod in enumerate(fon_listesi, start=1):
        print(f"[{_ts()}] ({i}/{len(fon_listesi)}) {kod} cekiliyor...")
        veri = fon_gecmisini_cek(kod, hedef_baslangic, hedef_bitis)
        tum_fon_verisi[kod] = veri
        print(f"  -> {len(veri)} gun bulundu.")
        time.sleep(BEKLEME_SANIYE)

    # --- birlesik (tum fonlarda gorulen) tarih listesi, yeniden eskiye ---
    tum_tarihler = set()
    for veri in tum_fon_verisi.values():
        tum_tarihler.update(veri.keys())
    yeni_tarihler = sorted(tum_tarihler, reverse=True)
    # sadece mevcut en eski tarihten KESINLIKLE eski olanlari al (cakismayi onle)
    yeni_tarihler = [t for t in yeni_tarihler if t < en_eski_tarih]

    if not yeni_tarihler:
        print(f"[{_ts()}] HATA: hicbir fon icin yeni tarih bulunamadi, "
              f"price_history.json degistirilmedi.")
        return

    print(f"[{_ts()}] Toplam {len(yeni_tarihler)} yeni (daha eski) trading gunu bulundu: "
          f"{yeni_tarihler[-1]} -> {yeni_tarihler[0]}")

    # --- her fon icin, yeni_tarihler sirasina gore fiyat dizisi olustur ---
    eksik_rapor = {}
    for kod in fon_listesi:
        veri = tum_fon_verisi.get(kod, {})
        mevcut_seri = fiyat_gecmisi["fiyatlar"].get(kod, [])
        son_bilinen_eski_fiyat = mevcut_seri[-1] if mevcut_seri else 0.0

        yeni_seri = []
        komsu = son_bilinen_eski_fiyat  # en yeni->eski giderken bir onceki (daha yeni) gunun fiyati
        for tarih in yeni_tarihler:
            fiyat = veri.get(tarih)
            if fiyat is None:
                fiyat = komsu  # eksik gun: komsu (bir yeni gunun) fiyatiyla doldur
                eksik_rapor[kod] = eksik_rapor.get(kod, 0) + 1
            yeni_seri.append(fiyat)
            komsu = fiyat
        fiyat_gecmisi["fiyatlar"][kod] = mevcut_seri + yeni_seri

    fiyat_gecmisi["tarihler"] = fiyat_gecmisi["tarihler"] + yeni_tarihler

    if eksik_rapor:
        print(f"[{_ts()}] {len(eksik_rapor)} fonda toplam eksik gun dolduruldu "
              f"(komsu fiyatla): {dict(list(eksik_rapor.items())[:10])}"
              f"{' ...' if len(eksik_rapor) > 10 else ''}")

    kaydet(fiyat_gecmisi, "price_history.json")
    print(f"[{_ts()}] price_history.json guncellendi. Yeni toplam gun sayisi: "
          f"{len(fiyat_gecmisi['tarihler'])} (once: {len(fiyat_gecmisi['tarihler']) - len(yeni_tarihler)})")


if __name__ == "__main__":
    main()
