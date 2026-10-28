# House Prices - Advanced Regression Techniques

**Kaggle Public Skoru: 0.12590**

Kaggle'ın [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) yarışması için uçtan uca bir makine öğrenmesi çözümü. Iowa, Ames şehrindeki konutların 79 açıklayıcı değişkenine dayanarak satış fiyatı tahmini yapılmaktadır.

## Sonuçlar

| Versiyon | Yöntem | CV RMSE (log) | Kaggle Public Skor |
|----------|--------|----------------|---------------------|
| v1 | Gradient Boosting (elle ayarlanmış parametreler) | 0.1345 | 0.13619 |
| v2 | v1 + Feature Engineering + Outlier temizliği | 0.1185 | 0.13499 |
| v3 | Optuna ile hiperparametre optimizasyonu (5 model) + Ağırlıklı Ensemble | 0.1093 | 0.12777 |
| v4 | Stacking (Ridge + Gradient Boosting + XGBoost → Linear meta-model) | 0.1099 | 0.12619 |
| v5 | v4 + Feature Selection (226 → 175 sütun, sıfır önemli sütunlar çıkarıldı) | 0.1098 | 0.12599 |
| v6 | v4 + Agresif Feature Selection (yalnızca en önemli 100 sütun) | — | 0.12724 (kötüleşti) |
| v7 | v5 + 5 model arasından otomatik en iyi kombinasyon araması (Ridge + XGBoost) | **0.1096** | **0.12590** ⭐ |

Değerlendirme metriği: tahmin ve gerçek `SalePrice` değerlerinin logaritmaları arasındaki RMSE (Root Mean Squared Error).

## Proje Yol Haritası

Proje şu adımlarla ilerlemiştir:

1. **Veri Keşfi (EDA)** — `train.csv` / `test.csv` yapısının incelenmesi, `data_description.txt` ile değişkenlerin (numeric / nominal / ordinal) sınıflandırılması.
2. **Eksik Veri Analizi ve Temizliği** — Eksik değerlerin "özellik yok" (`PoolQC`, `GarageType` vb.) ve "gerçek eksik veri" (`LotFrontage`, `Electrical` vb.) olarak ayrıştırılması; mahalle bazlı medyan, mod ve `0` ile doldurma stratejileri. Data leakage'dan kaçınmak için tüm istatistikler yalnızca train setinden hesaplanıp hem train hem test'e uygulanmıştır.
3. **Ordinal Encoding** — Kalite/durum bildiren 17 kategorik değişken (`ExterQual`, `BsmtExposure`, `Functional` vb.) sıralı sayısal değerlere dönüştürülmüştür.
4. **Nominal Encoding** — Kalan 26 kategorik değişken One-Hot Encoding ile dönüştürülmüş, train/test arasındaki kategori uyuşmazlıkları `reindex` ile hizalanmıştır.
5. **Hedef Değişken Dönüşümü** — `SalePrice` sağa çarpık dağılım gösterdiği için `log1p` dönüşümü uygulanmıştır (yarışmanın metriğiyle de örtüşür).
6. **Outlier Analizi** — `GrLivArea` çok yüksek olduğu halde `SalePrice`'ı düşük olan 2 aykırı ev (bilinen Ames veri seti outlier'ları) train setinden çıkarılmıştır.
7. **Feature Engineering** — `TotalSF`, `TotalBath`, `HouseAge`, `IsRemodeled` gibi türetilmiş özellikler eklenmiştir.
8. **Baseline Modeller** — Linear Regression, Random Forest, Gradient Boosting ile ilk karşılaştırmalar yapılmıştır.
9. **Hiperparametre Optimizasyonu** — Optuna ile 5 model (Ridge, Random Forest, Gradient Boosting, XGBoost, LightGBM) üzerinde 5-fold cross-validation ile sistematik arama yapılmıştır.
10. **Ensemble & Stacking** — En iyi modellerin ağırlıklı ortalaması ve Out-of-Fold (OOF) tahminlerine dayalı stacking (meta-model) denenmiştir.
11. **Feature Selection** — XGBoost feature importance'a göre hiç katkı sağlamayan sütunlar (226 → 175) çıkarılarak model sadeleştirilmiştir; daha agresif eşiklerin (ilk 100 sütun) performansı kötüleştirdiği gözlemlenmiştir.
12. **Model Kombinasyon Araması** — 5 modelin (Ridge, Random Forest, Gradient Boosting, XGBoost, LightGBM) tüm alt kümeleri (2'li, 3'lü, 4'lü, 5'li) sistematik olarak denenmiş, en iyi sonucu **Ridge + XGBoost** ikilisi vermiştir.

## Proje Yapısı

```
house-prices-advanced-regression-techniques/
│
├── data/
│   ├── raw/                          # Orijinal Kaggle verileri (değiştirilmez)
│   │   ├── train.csv
│   │   ├── test.csv
│   │   ├── sample_submission.csv
│   │   └── data_description.txt
│   └── processed/                    # Temizlenmiş / encode edilmiş veri
│       ├── train_processed.csv
│       └── test_processed.csv
│
├── notebooks/
│   └── notebooks/
│       ├── 01_eda.py                       # Veri keşfi, eksik veri temizliği, encoding, outlier temizliği
│       ├── 02_edaWithFE.py                 # Feature engineering + baseline model karşılaştırmaları
│       ├── 03_edaWithOptuna.py             # 5 model için Optuna hiperparametre optimizasyonu + ensemble
│       ├── 04_edaWithStacking.py           # Out-of-Fold tahminler + stacking (meta-model)
│       ├── 05_edaWithFeatureSelection.py   # Sıfır önemli sütunların çıkarılması (226 → 175)
│       ├── 06_edaWithFeatureSelection2.py  # Agresif feature selection denemesi (ilk 100 sütun)
│       └── 07_edaWithFullStacking.py       # 5 modelin tüm kombinasyonlarının otomatik taranması
│
├── submissions/                      # Kaggle'a gönderilen tahmin dosyaları
│   ├── submission_v1_gb.csv
│   ├── submission_v2_gb_fe.csv
│   ├── submission_v3_optuna_ensemble.csv
│   ├── submission_v4_stacking.csv
│   ├── submission_v5_feature_selection.csv
│   ├── submission_v6_top100_features.csv
│   └── submission_v7_full_stacking.csv
│
├── requirements.txt
└── README.md
```

## Kullanılan Yöntemler ve Kütüphaneler

- **Veri işleme:** pandas, numpy
- **Görselleştirme:** matplotlib, seaborn
- **Modelleme:** scikit-learn (Ridge, Random Forest, Gradient Boosting), XGBoost, LightGBM
- **Hiperparametre Optimizasyonu:** Optuna (Bayesian optimizasyon, 5-fold cross-validation ile)

## Nasıl Çalıştırılır

```bash
pip install -r requirements.txt
```

Ardından script'ler sırasıyla çalıştırılmalıdır (her biri bir öncekinin ürettiği veriye bağımlıdır):

```bash
python notebooks/notebooks/01_eda.py
python notebooks/notebooks/02_edaWithFE.py
python notebooks/notebooks/03_edaWithOptuna.py
python notebooks/notebooks/04_edaWithStacking.py
python notebooks/notebooks/05_edaWithFeatureSelection.py
python notebooks/notebooks/06_edaWithFeatureSelection2.py
python notebooks/notebooks/07_edaWithFullStacking.py
```

`01_eda.py` çalıştırıldığında `data/processed/` klasörüne temizlenmiş veri kaydedilir; sonraki script'ler bu işlenmiş veriyi okuyarak devam eder. En iyi sonucu (`v7`) üreten script `07_edaWithFullStacking.py`'dir.

## Öne Çıkan Öğrenimler

- **Outlier temizliği**, model seçiminden bağımsız olarak genel veri kalitesini iyileştirdi; özellikle ağaç tabanlı modellerin performansında belirgin fark yarattı.
- **Basit + regularize edilmiş bir model (Ridge)**, karmaşık ağaç modelleriyle (Random Forest, Gradient Boosting) yarışacak kadar güçlü çıktı — yüksek boyutlu ama nispeten doğrusal ilişkiler taşıyan bu veri setinde regularization'ın gücünü gösterdi.
- **Ensemble/stacking'de çeşitlilik, model sayısından daha önemli**: 5 modelin tüm kombinasyonları sistematik olarak denendiğinde, sadece **Ridge + XGBoost** ikilisi tüm 31 kombinasyon içinde en iyi sonucu verdi — Random Forest, Gradient Boosting ve LightGBM eklemek performansı artırmadı, çünkü bu modeller XGBoost ile büyük ölçüde örtüşen hatalar yapıyordu.
- **Feature selection'da "az önemli" ile "önemsiz" arasındaki fark kritik**: Hiç kullanılmayan (importance = 0) sütunları çıkarmak (226 → 175) performansı korurken, daha agresif bir eşikle (yalnızca en önemli 100 sütun) çıkarma yapmak gerçek bilgi kaybına yol açıp skoru kötüleştirdi.
- **Cross-validation skoru ile Kaggle Public skoru arasında sistematik bir fark** gözlemlendi (~0.01-0.02), bu da public leaderboard'un test setinin yalnızca bir alt kümesini yansıttığının ve tek bir validation ölçümüne aşırı güvenilmemesi gerektiğinin bir hatırlatıcısıdır.

## Veri Seti Hakkında

https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
