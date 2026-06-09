# Galaksi Segmentasyonunda Sürü Zekası ve Green AI Yaklaşımları

Bu proje, galaksi segmentasyonu problemlerine sürü zekası (GWO, PSO) tabanlı optimizasyon yöntemleri ve enerji verimli (Green AI) derin öğrenme stratejileri entegre eden bir araştırmadır.

## 🚀 Mimari
- **Ön İşleme:** Gri Kurt Optimizasyonu (GWO) ile adaptif eşikleme.
- **Optimizasyon:** Parçacık Sürü Optimizasyonu (PSO) ile hiper-parametre ayarı.
- **Model:** U-Net tabanlı segmentasyon.
- **Green AI:** Model budama (pruning) ve erken çıkış (early-exit) stratejileri (Gelecek çalışmalar).

## 📂 Dosya Yapısı

├── data/               # Veri setleri
├── models/             # U-Net mimarisi
├── optimizers/         # GWO, PSO, FA, WOA algoritmaları
├── utils/              # Dataset yardımcıları
├── main.py             # Ana orkestratör
└── requirements.txt    # Bağımlılıklar

## 🛠 Kurulum
pip install -r requirements.txt
python main.py

---

# Proje Teknik Dokümantasyonu

## 1. Giriş
Bu çalışma, yüksek hesaplama maliyetli galaksi segmentasyon işlemlerini, sürü zekası algoritmaları kullanarak optimize etmeyi ve enerji verimliliğini artırmayı hedefler.

## 2. Modüllerin Açıklaması

### `models/unet.py`
U-Net mimarisinin tanımlandığı katmandır. Gürültülü astronomik görüntülerde segmentasyon başarısını artırmak için optimize edilmiştir.

### `optimizers/gwo.py`
**Gri Kurt Optimizasyonu (GWO):** Görüntü ön işleme aşamasında kullanılır. Düşük kontrastlı galaksi bölgelerinin saptanması ve gürültüden arındırılması için adaptif eşikleme yapar.

### `optimizers/pso.py`
**Parçacık Sürü Optimizasyonu (PSO):** Modelin öğrenme oranı (LR), batch size ve kayıp fonksiyonu ağırlıklarını optimize eder. Deneme-yanılma yerine küresel arama yaparak enerji verimliliği sağlar.

### `utils/dataset.py`
`SyntheticGalaxyDataset` sınıfını barındırır. Astronomik görüntü verilerinin yüklenmesi, veri artırımı (augmentation) ve tensör dönüşümleri burada gerçekleştirilir.

## 3. Green AI Stratejileri
Sürü zekası algoritmaları ile sadece doğruluk değil, aynı zamanda karbon ayak izinin azaltılması hedeflenmektedir. Gelecek aşamalarda **Balina Optimizasyonu (WOA)** ile model budama (pruning) ve **Ateş Böceği Algoritması (FA)** ile erken çıkış mekanizmaları entegre edilecektir.

---
