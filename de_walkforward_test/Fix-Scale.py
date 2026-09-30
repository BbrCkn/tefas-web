import json, math
from pathlib import Path

p = Path("price_history_6y.json")
data = json.loads(p.read_text(encoding="utf-8"))
tarihler = data["tarihler"]  # desc: 0 en yeni
fiyatlar = data["fiyatlar"]

def duzelt():
    toplam = 0
    for kod, seri in fiyatlar.items():
        # desc: index 0 yeni, arttikca eski
        for i in range(len(seri)-1, 0, -1):
            eski = seri[i]
            yeni = seri[i-1]
            if not eski or not yeni or eski < 1e-9 or yeni < 1e-9:
                continue
            r = yeni / eski
            if r > 50 or r < 0.02:
                lg = math.log10(r)
                k = round(lg)
                if abs(lg - k) < 0.15 and abs(k) >= 2:
                    carpan = 10.0 ** k
                    # eski ve daha eski tum gunleri carpan ile carp
                    for j in range(i, len(seri)):
                        if seri[j]:
                            seri[j] *= carpan
                    toplam += 1
                    print(f"DUZELTILDI {kod} {tarihler[i-1]} r={r:.6g} log={lg:.2f} k={k} carpan={carpan} -> eski {eski:.4f} yeni {yeni:.4f} eski*carpan={eski*carpan:.4f}")
    return toplam

n = duzelt()
print(f"Toplam {n} duzeltme")
# ikinci kontrol
kalan = []
for kod, seri in fiyatlar.items():
    for i in range(len(seri)-1,0,-1):
        eski, yeni = seri[i], seri[i-1]
        if not eski or not yeni: continue
        r = yeni/eski if eski else 0
        if r>50 or (r>0 and r<0.02):
            kalan.append((kod, tarihler[i-1], r))
print(f"Kalan supheli: {len(kalan)}")
for x in kalan[:20]:
    print(x)

# kaydet
p.write_text(json.dumps(data, ensure_ascii=False, separators=(",",":")), encoding="utf-8")
print("Yazildi")
