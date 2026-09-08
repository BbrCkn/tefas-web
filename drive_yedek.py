"""
Pipeline basariyla calistiktan sonra repodaki kritik dosyalarin bir zip
yedegini, Babur'un kendi Google hesabinda calisan bir Apps Script web
adresine gonderir. Gercek Drive'a kaydetme islemi o script tarafinda,
Babur'un KENDI Google hesabi/kotasi ile yapilir -- boylece servis
hesaplarinin "depolama kotasi yok" kisitlamasina hic takilmayiz.

Ayni gun icin tek yedek / 10 gunluk pencere mantigi Apps Script
tarafinda (gdrive_yedek_alici.gs) uygulanir, burada sadece zip'leyip
gonderiyoruz.

Ortam degiskenleri (workflow'dan gelir):
  GAS_WEBHOOK_URL    -- Apps Script web app'in "/exec" ile biten adresi
  GAS_SECRET_TOKEN   -- gdrive_yedek_alici.gs icindeki GIZLI_ANAHTAR ile
                        AYNI olmasi gereken metin
"""
import os
import json
import base64
import zipfile
from datetime import datetime

import requests

YEDEKLENECEK_DOSYALAR = [
    "gunluk_pipeline.py",
    "puanlama_metrikleri.py",
    "requirements.txt",
    ".github/workflows/fetch-tefas.yml",
    "price_history.json",
    "portfoy.json",
    "fon_kazanc.json",
    "fon_listesi.json",
    "bekleyen_emirler.json",
    "bekleyen_valorler.json",
    "eklenecek_fonlar.json",
    "eksik_gun_doldur.json",
    "docs/index.html",
    "docs/data.json",
]


def yedek_zip_olustur(bugun_str):
    zip_adi = f"tefas-web-yedek-{bugun_str}.zip"
    with zipfile.ZipFile(zip_adi, "w", zipfile.ZIP_DEFLATED) as z:
        for dosya in YEDEKLENECEK_DOSYALAR:
            if os.path.exists(dosya):
                z.write(dosya)
            else:
                print(f"UYARI: yedeklenecek dosya bulunamadi, atlandi: {dosya}")
    return zip_adi


def main():
    bugun = datetime.now().strftime("%Y-%m-%d")
    zip_adi = yedek_zip_olustur(bugun)

    with open(zip_adi, "rb") as f:
        icerik_b64 = base64.b64encode(f.read()).decode("utf-8")

    yanit = requests.post(
        os.environ["GAS_WEBHOOK_URL"],
        json={
            "token": os.environ["GAS_SECRET_TOKEN"],
            "dosyaAdi": zip_adi,
            "icerikBase64": icerik_b64,
        },
        timeout=120,
    )
    yanit.raise_for_status()
    sonuc = yanit.json()
    if not sonuc.get("basarili"):
        raise RuntimeError(f"Drive yedekleme basarisiz: {sonuc.get('hata')}")
    print(f"Yedek Drive'a yuklendi: {zip_adi}")


if __name__ == "__main__":
    main()
