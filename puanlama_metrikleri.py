"""
puanlama_metrikleri.py
------------------------
2030.xlsm'deki Metrikler.bas modulunun (mdd, Otokorelasyon, Consistency,
EnBuyukGunOrani, PuanlamaMotoru) Python karsiligi.

Amac: Ayni fiyat verisini kullanarak VBA ile Python'un ayni sonuclari
uretip uretmedigini dogrulamak. Formuller VBA kodundan birebir cevrildi
(degisken adlari bile korundu, karsilastirmayi kolaylastirmak icin).
"""

import numpy as np


def puanlama_motoru(degerler: np.ndarray, yon: int, katsayi: float,
                     k: float, winsor: bool = True) -> np.ndarray:
    """
    PuanlamaMotoru'nun karsiligi: Winsorize (istege bagli) -> Z-score ->
    Sigmoid -> katsayi ile carpip dondur (Z sutununa toplama disaridan
    yapilir, bu fonksiyon sadece bir metrigin puanini uretir).
    """
    degerler = degerler.astype(float).copy()

    if winsor:
        p1, p99 = np.percentile(degerler, [1, 99])
        degerler = np.clip(degerler, p1, p99)

    ort = degerler.mean()
    std_sap = degerler.std(ddof=1)  # VBA: kareToplam / (n-1)

    if std_sap == 0:
        z_skor = np.zeros_like(degerler)
    else:
        z_skor = yon * (degerler - ort) / std_sap

    sigmoid = 1 / (1 + np.exp(-k * z_skor))
    return sigmoid * katsayi


def donemsel_getiriler(fiyat_matrisi: np.ndarray) -> dict:
    """
    Gunluk/haftalik/aylik/3 aylik/6 aylik/yillik getiri (%).
    fiyat_matrisi: (fon_sayisi, >=255) -- AK sutunundan itibaren, en az
    255 sutun (yillik icin 254 gun geriye bakmak gerekiyor).

    Formul (AD5 hucresindeki mantikla ayni):
        %getiri = (bugun - N_gun_once) / N_gun_once * 100
    """
    pencereler = {
        "gunluk": 1,
        "haftalik": 4,
        "aylik": 20,
        "uc_aylik": 62,
        "alti_aylik": 127,
        "yillik": 254,
    }
    bugun = fiyat_matrisi[:, 0]
    n_sutun = fiyat_matrisi.shape[1]
    sonuc = {}
    for ad, offset in pencereler.items():
        # Pencere, elimizdeki fiyat gecmisinden uzunsa (ornegin veri
        # henuz 255 gune ulasmadi ya da gecmiste bozuk bir gun temizlendi)
        # cakmak yerine en eski mevcut sutuna dus -- birkac gunluk pencere
        # gecici olarak biraz kisa hesaplanir, sistem hic durmaz.
        guvenli_offset = min(offset, n_sutun - 1)
        eski = fiyat_matrisi[:, guvenli_offset]
        with np.errstate(divide="ignore", invalid="ignore"):
            deger = np.where(eski != 0, (bugun - eski) / eski * 100, 0.0)
        deger = np.where(deger == -100, 0.0, deger)
        sonuc[ad] = deger
    return sonuc


def mdd(fiyat_matrisi: np.ndarray) -> np.ndarray:
    """
    Max Drawdown. fiyat_matrisi: (fon_sayisi, n) - AK..AK+n-1 sutunlari,
    en yeni fiyat solda (VBA'daki gibi). VBA dongusu eskiden yeniye
    gidiyordu (Step -1) ama tepe/mdd hesaplama yon-bagimsiz oldugu icin
    (kumulatif max/min mantigi), diziyi ters cevirip ayni sonucu aliyoruz.
    """
    n_fon, n = fiyat_matrisi.shape
    sonuc = np.zeros(n_fon)
    for i in range(n_fon):
        seri = fiyat_matrisi[i, ::-1]  # eskiden -> yeniye
        tepe = 0.0
        mdd_deger = 0.0
        for f in seri:
            if f > tepe:
                tepe = f
            if tepe > 0 and (tepe - f) / tepe > mdd_deger:
                mdd_deger = (tepe - f) / tepe
        sonuc[i] = mdd_deger
    return sonuc


def _log_getiri_serisi(fiyat_matrisi_np1: np.ndarray) -> np.ndarray:
    """fiyat_matrisi_np1: (fon_sayisi, n+1) -> (fon_sayisi, n) log getiri."""
    bugun = fiyat_matrisi_np1[:, :-1]
    dun = fiyat_matrisi_np1[:, 1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        getiri = np.where((dun > 0) & (bugun > 0), np.log(bugun / dun), 0.0)
    return getiri


def otokorelasyon(fiyat_matrisi_np1: np.ndarray) -> np.ndarray:
    """Lag-1 otokorelasyon (ham, isaretli - Abs() yok)."""
    getiri = _log_getiri_serisi(fiyat_matrisi_np1)  # (fon, n)
    n_fon, n = getiri.shape
    m = n - 1
    sonuc = np.zeros(n_fon)
    for i in range(n_fon):
        x = getiri[i, 1:]      # j = 2..n  (0-index: 1..n-1)
        y = getiri[i, :-1]     # j-1 = 1..n-1
        sx, sy = x.sum(), y.sum()
        sxy = (x * y).sum()
        sx2, sy2 = (x ** 2).sum(), (y ** 2).sum()
        payda = np.sqrt((m * sx2 - sx ** 2) * (m * sy2 - sy ** 2))
        sonuc[i] = 0.0 if payda == 0 else (m * sxy - sx * sy) / payda
    return sonuc


def consistency(fiyat_matrisi_np1: np.ndarray) -> np.ndarray:
    """Ortalama getiri / (std + 0.0001) + kazanan gun orani."""
    n_fon, n_plus1 = fiyat_matrisi_np1.shape
    n = n_plus1 - 1
    m = n - 1
    sonuc = np.zeros(n_fon)
    for i in range(n_fon):
        getiri = np.zeros(m)
        kazanan = 0
        for j in range(m):
            bugun = fiyat_matrisi_np1[i, j]
            dun = fiyat_matrisi_np1[i, j + 1]
            g = np.log(bugun / dun) if (dun > 0 and bugun > 0) else 0.0
            getiri[j] = g
            if g > 0:
                kazanan += 1
        ort = getiri.mean()
        std = getiri.std(ddof=1)
        if std == 0:
            sonuc[i] = kazanan / m
        else:
            sonuc[i] = (ort / (std + 0.0001)) + (kazanan / m)
    return sonuc


def en_buyuk_gun_orani(fiyat_matrisi_np1: np.ndarray) -> np.ndarray:
    """
    MAX(|log getiri|) / ORTALAMA(|log getiri|).
    NOT: VBA kodu burada m=n-1 getiri kullaniyor (Consistency ile ayni
    pencere), n degil -- ilk versiyonda bu gozden kacmisti, duzeltildi.
    """
    n_fon, n_plus1 = fiyat_matrisi_np1.shape
    m = n_plus1 - 2  # n-1
    dilim = fiyat_matrisi_np1[:, : m + 1]  # m+1 sutun -> m getiri
    getiri = np.abs(_log_getiri_serisi(dilim))
    max_abs = getiri.max(axis=1)
    toplam_abs = getiri.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sonuc = np.where(toplam_abs == 0, 0.0, max_abs / (toplam_abs / m))
    return sonuc


def tema_olustur(fiyat_91: np.ndarray, kodlar: list, valorler: np.ndarray) -> list:
    """
    TemaOluştur.bas'in Python karsiligi. K-means benzeri kume atama:
    - Sadece valoru sayisal ve >0 olan fonlar kumeleniyor
    - 90 gunluk getiri, fon bazinda standardize ediliyor (z-score)
    - 8 baslangic merkezi = ilk 8 fonun getiri vektoru
    - 15 iterasyon k-means (mesafe icin sadece ilk 30 gun kullanilir,
      merkez guncellemesi 90 gunun tamami uzerinden)
    - Buyuk gruplari (>28 fon) en fazla 10 kez ikiye bolme
    - Buyuk gruplari (>32 fon) en fazla 80 kez kucultme (tek tek tasima)

    fiyat_91: (fon_sayisi, 91) -- AK..AK+90 sutunlari (tum evren, 203 fon)
    kodlar: fon kodlari listesi (fiyat_91 ile ayni sirada)
    valorler: C sutunundaki valor degerleri (ayni sirada)

    Donus: [(fon_kodu, "grupNo.sira"), ...] -- sadece kumelenen fonlar icin
    """
    gecerli = [i for i in range(len(kodlar))
               if valorler[i] is not None and valorler[i] > 0]
    fon_sayisi = len(gecerli)
    if fon_sayisi == 0:
        return []

    # --- getiri: (fon_sayisi, 90), fon bazinda standardize ---
    getiri = np.zeros((fon_sayisi, 90))
    for idx, i in enumerate(gecerli):
        p1 = fiyat_91[i, 0:90]
        p2 = fiyat_91[i, 1:91]
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.where(p2 != 0, p1 / p2 - 1, 0.0)
        ort = g.mean()
        v = np.sqrt(((g - ort) ** 2).mean())
        if v < 0.00001:
            v = 1.0
        getiri[idx] = (g - ort) / v

    # --- k-means, k=8, 15 iterasyon, mesafe icin ilk 30 gun ---
    K = 8
    merkez = getiri[:K].copy()
    grup_no = np.zeros(fon_sayisi, dtype=int)

    for _ in range(15):
        fark = getiri[:, None, :30] - merkez[None, :, :30]
        mesafe = (fark ** 2).sum(axis=2)
        grup_no = mesafe.argmin(axis=1)

        yeni_merkez = np.zeros_like(merkez)
        for k in range(K):
            uyeler = getiri[grup_no == k]
            if len(uyeler) > 0:
                yeni_merkez[k] = uyeler.mean(axis=0)
            else:
                yeni_merkez[k] = merkez[k]
        merkez = yeni_merkez

    # --- bolme: en buyuk grup > 28 ise ikiye bol (en fazla 10 kez) ---
    max_grup = K - 1  # 0-index; VBA'da maxGrup=8 (1-index) -> burada son index 7
    for _ in range(10):
        sayilar = np.bincount(grup_no, minlength=max_grup + 1)
        en_buyuk = sayilar.argmax()
        en_buyuk_sayi = sayilar[en_buyuk]
        if en_buyuk_sayi <= 28 or max_grup >= 9:
            break
        uyeler = np.where(grup_no == en_buyuk)[0]
        if len(uyeler) < 2:
            break
        # en birbirinden uzak iki uye (ilk 30 gun uzerinden)
        alt = getiri[uyeler][:, :30]
        fark = alt[:, None, :] - alt[None, :, :]
        mesafe = (fark ** 2).sum(axis=2)
        a_idx, b_idx = np.unravel_index(np.argmax(mesafe), mesafe.shape)
        t1, t2 = uyeler[a_idx], uyeler[b_idx]
        m1, m2 = getiri[t1, :30], getiri[t2, :30]

        max_grup += 1
        for tt in uyeler:
            d1 = ((getiri[tt, :30] - m1) ** 2).sum()
            d2 = ((getiri[tt, :30] - m2) ** 2).sum()
            if d2 < d1:
                grup_no[tt] = max_grup

    # --- dengeleme: en buyuk grup > 32 ise tek tek en kucuge tasi ---
    for _ in range(80):
        sayilar = np.bincount(grup_no, minlength=max_grup + 1)
        dolu = np.where(sayilar > 0)[0]
        b_max = dolu[np.argmax(sayilar[dolu])]
        k_min = dolu[np.argmin(sayilar[dolu])]
        if sayilar[b_max] <= 32:
            break
        ilk_uye = np.where(grup_no == b_max)[0][0]
        grup_no[ilk_uye] = k_min

    # --- yaz: "grupNo.sira" (grupNo 1-index, VBA'daki gibi) ---
    sayac = {}
    sonuc = []
    for idx, i in enumerate(gecerli):
        g = int(grup_no[idx]) + 1  # 1-index'e cevir
        sayac[g] = sayac.get(g, 0) + 1
        sonuc.append((kodlar[i], f"{g}.{sayac[g]:02d}"))
    return sonuc


def sharpe(fiyat_matrisi_np1: np.ndarray, risksiz_yillik: float) -> np.ndarray:
    """
    KL formulunun karsiligi:
    (ortalama_log_getiri - ((1+rf)^(1/n) - 1)) / std_orneklem * sqrt(n)
    fiyat_matrisi_np1: (fon_sayisi, n+1) -- AK..AK+n sutunlari.
    """
    getiri = _log_getiri_serisi(fiyat_matrisi_np1)  # (fon, n)
    n = getiri.shape[1]
    ort = getiri.mean(axis=1)
    std = getiri.std(axis=1, ddof=1)
    esik = (1 + risksiz_yillik) ** (1 / n) - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        sonuc = np.where(std == 0, 0.0, (ort - esik) / std * np.sqrt(n))
    return sonuc
