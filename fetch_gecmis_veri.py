"""
fetch_gecmis_veri.py (revize) -- TEK SEFERLIK / gerektiginde tekrar calistirilir.

Mevcut price_history dosyasinin en eski tarihinden GERIYE dogru, hedef yil sayisina
ulasana kadar TEFAS'tan fiyat ceker ve ayri bir dosyaya yazar. Girdiyi DEGISTIRMEZ.

Orijinal versiyondan farklar:
  * --yil ile kac yillik toplam gecmis istendigi secilir (varsayilan 6).
    (Eski sabit HEDEF_TAKVIM_GUNU=400 sadece ~1.1 yil ekliyordu.)
  * Cikti ayri dosyaya yazilir (varsayilan price_history_6y.json); uretim dosyasi bozulmaz.
  * Yillik parcalara bolunur, her parca bitince ara dosyaya (gecmis_cache.json) kaydedilir.
    Hata olursa ayni komutla kaldigi yerden devam eder.
  * Fon ilk gercek fiyatindan ONCE 0.0 kalir (mevcut dosyanin kurali: var olmayan fon = 0,
    hep en eski uctadir). Eski kod bu gunleri fonun ilk fiyatiyla doldurup hayali duz seri
    uretiyordu.
  * Eksik gun, BIR ONCEKI (daha eski) bilinen fiyatla doldurulur; eski kod daha yeni gunun
    fiyatini kullaniyordu (kucuk ileri bakis).
  * Sadece girdi dosyasindaki fonlar islenir (temizlenmis evren icin --girdi ile
    price_history_temiz.json verilebilir). Evrende olmayan fon eklenmez.
  * Olcek kaymasi (TI6/GA1 tipi) tespit edilip raporlanir; --olcek-duzelt ile duzeltilir.
"""
import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path

KOK = Path(__file__).parent


def yukle(dosya):
    with open(KOK / dosya, encoding="utf-8") as f:
        return json.load(f)


def kaydet(veri, dosya):
    with open(KOK / dosya, "w", encoding="utf-8") as f:
        json.dump(veri, f, ensure_ascii=False, separators=(",", ":"))


def _ts():
    return dt.datetime.now().strftime("%H:%M:%S")


def cek_parca(baslangic, bitis, kodlar):
    """[baslangic, bitis] icin TUM fonlari tek cagriyla ceker; sadece `kodlar`i tutar.
    Donus: {kod: {YYYY-MM-DD: fiyat}}"""
    from pytefas import Crawler
    df = Crawler().fetch(baslangic, bitis, columns="info", kind="YAT")
    sonuc = {}
    for k in json.loads(df.to_json(orient="records")):
        kod, fiyat = k.get("fund_code"), k.get("price")
        tarih = str(k.get("date", ""))[:10]
        if kod in kodlar and tarih and fiyat is not None:
            sonuc.setdefault(kod, {})[tarih] = float(fiyat)
    return sonuc


def parcalari_hazirla(en_eski, hedef_baslangic, parca_gun):
    parcalar, bitis = [], en_eski - dt.timedelta(days=1)
    while bitis >= hedef_baslangic:
        bas = max(bitis - dt.timedelta(days=parca_gun - 1), hedef_baslangic)
        parcalar.append((bas, bitis))
        bitis = bas - dt.timedelta(days=1)
    return parcalar


def seri_olustur(veri, asc_tarihler):
    """veri: {tarih: fiyat}. Fon ilk fiyatindan once 0.0; sonrasinda eksik gun = onceki fiyat."""
    ilk = min(veri) if veri else None
    son, seri = None, []
    for t in asc_tarihler:
        if t in veri:
            son = veri[t]
            seri.append(son)
        elif ilk is not None and t > ilk and son is not None:
            seri.append(son)
        else:
            seri.append(0.0)
    return seri


def olcek_kontrol(tarihler_desc, fiyatlar, duzelt):
    """Gunluk oran >50x veya <1/50 olan sicramalari bulur. 10'un kuvvetine yakinsa
    (|k|>=2) ve duzelt=True ise ESKI fiyatlari olcekler. Dondurur: bulgu listesi."""
    bulgular = []
    for kod, seri in fiyatlar.items():
        # seri yeni->eski; eskiden yeniye tara
        for i in range(len(seri) - 1, 0, -1):
            eski, yeni = seri[i], seri[i - 1]
            if not eski or not yeni:
                continue
            r = yeni / eski
            if r > 50 or r < 0.02:
                lg = math.log10(r)
                k = round(lg)
                yakin = abs(lg - k) < 0.15 and abs(k) >= 2
                bulgular.append((kod, tarihler_desc[i - 1], r, yakin))
                if yakin and duzelt:
                    carpan = 10.0 ** k
                    for j in range(i, len(seri)):
                        if seri[j]:
                            seri[j] *= carpan
    return bulgular


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yil", type=float, default=6.0, help="toplam hedef gecmis (yil)")
    ap.add_argument("--girdi", default="price_history.json")
    ap.add_argument("--cikti", default="price_history_6y.json")
    ap.add_argument("--cache", default="gecmis_cache.json")
    ap.add_argument("--parca-gun", type=int, default=365)
    ap.add_argument("--olcek-duzelt", action="store_true")
    a = ap.parse_args()

    mevcut = yukle(a.girdi)
    kodlar = set(mevcut["fiyatlar"])
    en_yeni = dt.date.fromisoformat(mevcut["tarihler"][0])
    en_eski = dt.date.fromisoformat(mevcut["tarihler"][-1])
    hedef = en_yeni - dt.timedelta(days=int(a.yil * 365.25))
    if hedef >= en_eski:
        print(f"[{_ts()}] Girdi zaten hedefi karsiliyor ({en_eski} <= {hedef}).")
        return
    print(f"[{_ts()}] Mevcut: {en_eski} -> {en_yeni}; hedef baslangic: {hedef}; "
          f"{len(kodlar)} fon")

    cache_yolu = KOK / a.cache
    cache = yukle(a.cache) if cache_yolu.exists() else {"biten": [], "veri": {}}
    for bas, bit in parcalari_hazirla(en_eski, hedef, a.parca_gun):
        anahtar = f"{bas}_{bit}"
        if anahtar in cache["biten"]:
            print(f"[{_ts()}] {anahtar} zaten var, atlandi")
            continue
        print(f"[{_ts()}] Cekiliyor: {bas} -> {bit}")
        try:
            parca = cek_parca(bas.isoformat(), bit.isoformat(), kodlar)
        except Exception as e:  # kismi sonuc kaydedildi; yeniden calistirinca devam eder
            print(f"[{_ts()}] HATA ({bas} -> {bit}): {e!r}. Yeniden calistirin.")
            sys.exit(1)
        for kod, d in parca.items():
            cache["veri"].setdefault(kod, {}).update(d)
        cache["biten"].append(anahtar)
        kaydet(cache, a.cache)
        print(f"[{_ts()}]   {sum(len(d) for d in parca.values())} kayit, {len(parca)} fon")

    en_eski_s = en_eski.isoformat()
    yeni = sorted({t for d in cache["veri"].values() for t in d if t < en_eski_s},
                  reverse=True)
    if not yeni:
        print(f"[{_ts()}] HATA: yeni tarih yok, cikti yazilmadi.")
        sys.exit(1)
    asc = yeni[::-1]
    fiyatlar = {}
    for kod, eski_seri in mevcut["fiyatlar"].items():
        fiyatlar[kod] = list(eski_seri) + seri_olustur(cache["veri"].get(kod, {}), asc)[::-1]  # yeni->eski
    tarihler = mevcut["tarihler"] + yeni
    assert all(len(v) == len(tarihler) for v in fiyatlar.values())

    bulgular = olcek_kontrol(tarihler, fiyatlar, a.olcek_duzelt)
    for kod, t, r, yakin in bulgular:
        durum = ("DUZELTILDI" if (yakin and a.olcek_duzelt) else
                 "duzeltilebilir (--olcek-duzelt)" if yakin else "MANUEL BAK")
        print(f"[{_ts()}] OLCEK? {kod} {t} oran={r:.6g} -> {durum}")

    veri_yok = [k for k in kodlar if k not in cache["veri"]]
    if veri_yok:
        print(f"[{_ts()}] Yeni donemde hic veri olmayan {len(veri_yok)} fon: {veri_yok[:15]}")
    kaydet({"tarihler": tarihler, "fiyatlar": fiyatlar}, a.cikti)
    print(f"[{_ts()}] Yazildi: {a.cikti} -- {len(tarihler)} gun "
          f"({tarihler[-1]} -> {tarihler[0]}), {len(fiyatlar)} fon")


if __name__ == "__main__":
    main()
