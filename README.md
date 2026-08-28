# TEFAS Web Projesi — Prototip 1: Veri Çekme Testi

## Bu ne işe yarar?
`fetch_tefas.py`, TEFAS'ın resmi API'sinden günün fon verilerini çekip
`tefas_gunluk.json` dosyasına kaydeder. Bu, 2030.xlsm'deki "Update"
butonunun veri indirme kısmının web/otomatik karşılığıdır.

Python bilmenize gerek yok — aşağıdaki adımlar sadece tarayıcı üzerinden
tıklamalarla yapılıyor.

## Neden burada değil de Render'da test ediyoruz?
Şu an kodu hazırladığım ortamın güvenlik nedeniyle internet erişimi
kısıtlı (sadece belirli teknik siteler açık, TEFAS dahil değil).
Gerçek barındırma ortamında (Render) bu kısıtlama olmayacak — bu yüzden
asıl "çalışıyor mu?" testini orada yapacağız.

## Adım adım: Render'da ilk test

1. **render.com** adresine gidin, ücretsiz hesap açın (GitHub hesabınızla
   giriş yapabilirsiniz, yoksa e-posta ile de olur).
2. Bana bu adımdan sonra haber verin — GitHub hesabınız var mı yok mu
   söyleyin, ona göre en kolay yolu birlikte seçelim (kod deposu
   oluşturma kısmını ben hazırlayabilirim).
3. Render'da "New Web Service" (Yeni Web Servisi) seçeneğine tıklanacak
   ve bu proje dosyaları bağlanacak.
4. Render otomatik olarak `requirements.txt`'i kurup `fetch_tefas.py`'yi
   çalıştıracak.
5. Sonuç ekranda: kaç fon için veri çekildiğini ve ilk birkaç fonun
   fiyatını göreceğiz. Bu, planın en kritik varsayımını (TEFAS
   API'sine bulut ortamından erişilebiliyor mu) doğrulayacak.

## Sırada ne var?
Bu adım başarılı olursa:
- Veri çekme mantığını her gün 10:30 sonrası otomatik çalışacak hale
  getireceğiz (Render'ın zamanlanmış görev özelliğiyle).
- Ardından 2030.xlsm'deki hesaplama zincirini (Rutin, Metrikler,
  TemaOluştur, Backtest) Python'a taşımaya başlayacağız.
- En son, telefon/tabletten açılabilen web tablosunu ekleyeceğiz.

Bu dosyayı ve `fetch_tefas.py`'yi siz düzenlemeyeceksiniz — ben
geliştirmeye devam edeceğim, siz sadece test sonuçlarını
onaylayacaksınız.
