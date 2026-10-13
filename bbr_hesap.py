"""
bbr_hesap.py
------------
BBR (portfoyun bilesik "fon payi" fiyati) hesabi -- Excel'deki Rutin makrosunun
"5. PERFORMANS HESAPLAMA (G221/G222)" blogunun web karsiligi.

Excel:   pay_bugun   = pay_dun + (V_bugun - V_dun - gunluk_kazanc) / BBR_fiyat_dun
         BBR_fiyat   = V_bugun / pay_bugun
(V = portfoyun TL degeri. V_bugun - V_dun - kazanc = o gunku net AL/SAT akisi.)

Durum bbr_durum.json'da tutulur ve ayni gun birden fazla calisma guvenlidir
(ayni tarihte hep "onceki gun kapanisi" baz alinir).
"""
import json
from pathlib import Path

KOK = Path(__file__).parent


def _yukle(ad):
    yol = KOK / ad
    if not yol.is_file():
        return None
    with open(yol, encoding="utf-8") as f:
        return json.load(f)


def _kaydet(veri, ad):
    with open(KOK / ad, "w", encoding="utf-8") as f:
        json.dump(veri, f, ensure_ascii=False, indent=1)


def excelden_baslat(fiyat_gecmisi: dict) :
    """Tek seferlik: bbr_excel_baslangic.json'daki Excel BBR serisini
    price_history'ye TARIHE gore hizalayip yazar ve bbr_durum.json'u kurar.
    Dosya yoksa ya da durum zaten varsa hicbir sey yapmaz."""
    if (KOK / "bbr_durum.json").is_file():
        return _yukle("bbr_durum.json")
    ex = _yukle("bbr_excel_baslangic.json")
    if not ex:
        return None
    tarihler = fiyat_gecmisi["tarihler"]
    son = ex["son_tarih"]
    if son not in tarihler:
        print(f"[BBR] UYARI: Excel baslangic tarihi {son} fiyat gecmisinde yok, atlandi.")
        return None
    j = tarihler.index(son)
    seri = fiyat_gecmisi["fiyatlar"]["BBR"]
    n = 0
    for i, p in enumerate(ex["fiyatlar"]):
        if j + i < len(seri):
            seri[j + i] = float(p)
            n += 1
    print(f"[BBR] Excel serisinden {n} gun yazildi ({son} ve oncesi).")
    durum = {
        "tarih": son,
        "bugun": ex["bugun"],
        "onceki": ex["onceki"],
    }
    _kaydet(durum, "bbr_durum.json")
    return durum


def gunluk_hesapla(durum: dict, tarih: str, deger_bugun: float, gunluk_kazanc_toplam: float):
    """Verilen tarih icin BBR fiyatini hesaplar, durumu gunceller.
    Donus: (yeni_fiyat | None, durum)."""
    if durum is None:
        return None, durum
    if durum["tarih"] == tarih:
        baz = durum["onceki"]              # ayni gun tekrar calisma
    elif durum["tarih"] < tarih:
        baz = durum["bugun"]               # yeni gun: dunun kapanisi
        durum = {"tarih": tarih, "onceki": baz, "bugun": baz}
    else:
        return None, durum                 # eski tarih -- dokunma
    if deger_bugun <= 0 or baz["pay"] <= 0 or baz["fiyat"] <= 0:
        return None, durum
    net_akis = deger_bugun - baz["deger"] - gunluk_kazanc_toplam
    pay = baz["pay"] + net_akis / baz["fiyat"]
    if pay <= 0:
        return None, durum
    fiyat = deger_bugun / pay
    durum["bugun"] = {"pay": pay, "deger": deger_bugun, "fiyat": fiyat}
    durum["tarih"] = tarih
    return fiyat, durum


def kaydet_durum(durum):
    if durum is not None:
        _kaydet(durum, "bbr_durum.json")
