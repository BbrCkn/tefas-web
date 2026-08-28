"""
fetch_tefas.py
----------------
Amaç: TEFAS'in resmi JSON API'sinden gunun fon verilerini cekip
bir JSON dosyasina kaydetmek. 2030.xlsm'deki Update_Click / Manuel_Click
makrolarinin "veri indirme" kismina karsilik gelir.

Bu betik Babur'un Python bilmesine gerek kalmadan calisacak sekilde
tasarlandi: tek yapmasi gereken bu dosyayi (veya onun uzerine kurulu
web servisini) calistirmak. Kod, olabildigince az bagimlilikla,
anlasilir Turkce yorum satirlariyla yazildi.

Kullanilan kutuphane: pytefas (TEFAS'in resmi, 2026'da yenilenen
Next.js tabanli API'sini kullanir; giris/API anahtari gerekmez).
"""

import json
from datetime import date, datetime, timedelta

from pytefas import Crawler, TefasAPIError, TefasRateLimitError


def bugunun_tarihi_veya_son_is_gunu() -> str:
    """TEFAS hafta sonu veri vermez; hafta sonuysa son is gununu dondurur."""
    bugun = date.today()
    # 5 = Cumartesi, 6 = Pazar (Python'da Pazartesi=0)
    while bugun.weekday() >= 5:
        bugun = bugun - timedelta(days=1)
    return bugun.strftime("%Y-%m-%d")


def fon_verilerini_cek(tarih: str | None = None) -> list[dict]:
    """
    TEFAS'tan o gune ait tum yatirim fonu (YAT) verilerini ceker.

    Donen her kayit: fund_code, fund_name, price, shares_outstanding,
    investor_count, portfolio_size gibi alanlar icerir
    (2030.xlsm'deki A/B/AK sutunlarinin karsiligi).
    """
    if tarih is None:
        tarih = bugunun_tarihi_veya_son_is_gunu()

    tefas = Crawler()
    df = tefas.fetch(tarih, columns="info", kind="YAT")

    # DataFrame'i normal bir Python listesine (JSON'a hazir) ceviriyoruz
    kayitlar = json.loads(df.to_json(orient="records", date_format="iso"))
    return kayitlar


def kaydet(kayitlar: list[dict], dosya_adi: str = "tefas_gunluk.json") -> None:
    """Cekilen veriyi okunabilir bir JSON dosyasina yazar."""
    cikti = {
        "cekilme_zamani": datetime.now().isoformat(timespec="seconds"),
        "fon_sayisi": len(kayitlar),
        "fonlar": kayitlar,
    }
    with open(dosya_adi, "w", encoding="utf-8") as f:
        json.dump(cikti, f, ensure_ascii=False, indent=2)


def main() -> None:
    print("TEFAS'tan veri cekiliyor...")
    try:
        kayitlar = fon_verilerini_cek()
    except TefasRateLimitError:
        print("HATA: TEFAS istek limitine takildik. Birazdan tekrar deneyin.")
        return
    except TefasAPIError as e:
        print(f"HATA: TEFAS API hatasi: {e}")
        return

    if not kayitlar:
        print("UYARI: Hic veri donmedi (bugun is gunu olmayabilir ya da "
              "veri henuz yayinlanmamis olabilir).")
        return

    kaydet(kayitlar)
    print(f"Basarili: {len(kayitlar)} fon icin veri cekildi ve "
          f"tefas_gunluk.json dosyasina kaydedildi.")
    print("Ornek ilk 3 kayit:")
    for kayit in kayitlar[:3]:
        print(" -", kayit.get("fund_code"), kayit.get("fund_name"),
              kayit.get("price"))


if __name__ == "__main__":
    main()
