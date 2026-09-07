"""
Pipeline basariyla calistiktan sonra repodaki kritik dosyalarin bir zip
yedegini Google Drive'a yukler.

- Ayni gun icin sadece TEK bir yedek dosyasi tutulur (dosya adi tarihe
  gore sabit: tefas-web-yedek-YYYY-MM-DD.zip). Ayni gun icinde pipeline
  birden fazla kez calisirsa, o gune ait onceki yedek silinip yenisiyle
  degistirilir -- yani o gunun EN SON basarili calismasinin yedegi kalir.
- Son 10 gunden eski yedekler otomatik silinir (dinamik pencere,
  2030.xlsm'deki yedekleme mantiginin karsiligi).
- Bu script sadece "Gunluk pipeline'i calistir" adimi BASARILI oldugunda
  calisir (workflow'daki adim sirasi/varsayilan GitHub Actions davranisi
  sayesinde -- onceki adim hata verirse bu adim hic calismaz).

Ortam degiskenleri (workflow'dan gelir):
  GDRIVE_CREDENTIALS -- servis hesabinin JSON kimlik bilgisi (GitHub secret)
  GDRIVE_FOLDER_ID   -- hedef Drive klasorunun ID'si
"""
import os
import json
import zipfile
from datetime import datetime, timedelta

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

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

TUTULACAK_GUN_SAYISI = 10
YEDEK_ADI_ONEKI = "tefas-web-yedek-"


def yedek_zip_olustur(bugun_str):
    zip_adi = f"{YEDEK_ADI_ONEKI}{bugun_str}.zip"
    with zipfile.ZipFile(zip_adi, "w", zipfile.ZIP_DEFLATED) as z:
        for dosya in YEDEKLENECEK_DOSYALAR:
            if os.path.exists(dosya):
                z.write(dosya)
            else:
                print(f"UYARI: yedeklenecek dosya bulunamadi, atlandi: {dosya}")
    return zip_adi


def drive_servisi():
    creds_dict = json.loads(os.environ["GDRIVE_CREDENTIALS"])
    creds = service_account.Credentials.from_service_account_info(
        creds_dict, scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=creds)


def yedekleri_listele(servis, klasor_id):
    sonuc = servis.files().list(
        q=(f"'{klasor_id}' in parents and trashed = false "
           f"and name contains '{YEDEK_ADI_ONEKI}'"),
        fields="files(id, name)",
        pageSize=1000,
    ).execute()
    return sonuc.get("files", [])


def main():
    klasor_id = os.environ["GDRIVE_FOLDER_ID"]
    bugun = datetime.now().strftime("%Y-%m-%d")
    zip_adi = yedek_zip_olustur(bugun)

    servis = drive_servisi()
    mevcut = yedekleri_listele(servis, klasor_id)

    # Ayni gune ait ONCEKI yedek varsa sil -- o gunun sadece EN SON
    # calismasinin yedegi kalsin.
    for f in mevcut:
        if f["name"] == zip_adi:
            servis.files().delete(fileId=f["id"]).execute()
            print(f"Ayni gunun eski yedegi silindi: {f['name']}")

    medya = MediaFileUpload(zip_adi, mimetype="application/zip")
    servis.files().create(
        body={"name": zip_adi, "parents": [klasor_id]},
        media_body=medya,
        fields="id",
    ).execute()
    print(f"Yedek Drive'a yuklendi: {zip_adi}")

    # Dinamik 10 gunluk pencere: daha eski yedekleri sil.
    sinir_tarih = datetime.now() - timedelta(days=TUTULACAK_GUN_SAYISI)
    for f in yedekleri_listele(servis, klasor_id):
        tarih_str = f["name"].replace(YEDEK_ADI_ONEKI, "").replace(".zip", "")
        try:
            dosya_tarihi = datetime.strptime(tarih_str, "%Y-%m-%d")
        except ValueError:
            continue
        if dosya_tarihi < sinir_tarih:
            servis.files().delete(fileId=f["id"]).execute()
            print(f"10 gunden eski yedek silindi: {f['name']}")


if __name__ == "__main__":
    main()
