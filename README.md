# Stock Price Forecasting with PyTorch (LSTM & GRU)

Zaman serisi analizi, stationarity teşhisi ve PyTorch tabanlı LSTM / GRU derin öğrenme mimarileri ile Amazon (AMZN) hisse senedi getiri ve fiyat tahminleme deneyi.

---

## 📌 Proje Akışı ve Metodolojik Yolculuk

Bu proje, hisse senedi zaman serisi tahmininde sıkça düşülen metodolojik hataları adım adım tespit edip düzelten kapsamlı bir makine öğrenmesi çalışmasıdır:

```
1. Problem Tanımı: AMZN Kapanış Fiyatı Tahmini
   │
   ▼
2. İlk Deney (Raw Close & Sliding Window):
   • Model: GRU (RMSE: $10.09)
   • Görünüşte "başarılı" kabul edilen modelin sorgulanması
   │
   ▼
3. Naive Baseline Tespiti:
   • Persistence Baseline (P_t = P_{t-1}) -> RMSE: $3.73
   • Derin öğrenme modelinin ($10.09), bir önceki günü kopyalayan basit baseline'dan ($3.73) çok daha kötü olduğu anlaşıldı.
   │
   ▼
4. Zaman Serisi Teşhisleri & Stationarity Analizi:
   • ADF & KPSS Testleri: Ham Close serisinin durağan olmadığı (non-stationary, p=0.9780) tespit edildi.
   • Non-stationary fiyat serisi, sinir ağlarının gerçek örüntü öğrenmek yerine sadece gecikmeli seviye takibi (lagged persistence) yapmasına neden oluyordu.
   │
   ▼
5. Hedef Dönüşümü (Target Transformation):
   • Log Return: r_t = ln(Close_t / Close_{t-1})
   • ADF (p=0.0000) ve KPSS (p=0.1000) ile durağanlık doğrulandı.
   │
   ▼
6. Çoklu Baseline Kurgusu (Aynı Test Döneminde):
   • Zero Return (r_t = 0)
   • Previous Return (r_t = r_{t-1})
   • Linear Regression (Gecikmeli getiriler: Lag 1, 2, 3, 5, 10)
   │
   ▼
7. PyTorch Derin Öğrenme Karşılaştırması:
   • LSTM vs GRU (lookback=20, hidden_size=32, num_layers=2)
   • Validation Loss takibi, Early Stopping ve `copy.deepcopy` ile best checkpoint restorasyonu.
   │
   ▼
8. Fiyat Uzayında One-Step-Ahead Rekonstrüksiyon:
   • Predicted_Close_t = Actual_Close_{t-1} * exp(Predicted_LogReturn_t)
   • Price RMSE ile Naive Baseline ($3.7266) kıyaslaması.
```

---

## 📊 Final Sonuçlar

Bütün modeller **aynı test örnekleri ($N = 604$)** ve **aynı zaman dilimi** üzerinde, veri sızıntısı (future leakage) olmaksızın test edilmiştir.

### 1. Log Return Tahmin Performansı

| Model | N | Validation RMSE | Test RMSE | Test MAE | Directional Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Zero Return** | 604 | 0.02478 | 0.01945 | 0.01399 | - |
| **Previous Return** | 604 | 0.03494 | 0.02803 | 0.02049 | 48.68% |
| **Linear Regression (Lags)** | 604 | 0.02486 | **0.01940** | **0.01396** | **51.99%** |
| **LSTM** | 604 | 0.02480 | 0.01951 | 0.01406 | 47.02% |
| **GRU** | 604 | 0.02479 | 0.01947 | 0.01402 | 47.85% |

### 2. Price-Space Evaluation (One-Step-Ahead Fiyat Hatası)

| Model | Price RMSE ($) |
| :--- | :---: |
| **Naive Price Baseline ($P_t = P_{t-1}$)** | **$3.726560** |
| **GRU** | **$3.729627** |
| **LSTM** | **$3.736202** |

---

## 🔍 Kritik Değerlendirme ve Sonuç

* **Model Karşılaştırması:** GRU modeli ($Price\_RMSE = \$3.7296$), LSTM modelinden ($Price\_RMSE = \$3.7362$) çok az daha iyi performans göstermiştir.
* **Baseline Kıyaslaması:** Hem LSTM hem de GRU modelleri, en basit referans noktaları olan **Zero Return** ($Test\_RMSE = 0.01945$) ve **Naive Price Persistence** ($Price\_RMSE = \$3.7266$) üzerinde anlamlı bir prediktif iyileşme (predictive improvement) **sağlayamamıştır**.
* **Çıkarım:** Bu projenin çıktısı "karlı/başarılı bir borsa tahmin botu" değil; finansal zaman serilerinde ham fiyat hedeflemenin yarattığı illüzyonu kıran, durağanlık analizini doğru uygulayan, validation/test sınırlarını katı şekilde koruyan ve derin öğrenme modellerini adil baseline'larla kıyaslayan **metodolojik olarak doğru ve savunulabilir bir makine öğrenmesi deneyidir**.

---

## 📁 Proje Dosya Yapısı

```
stock-price-pytorch/
├── data/
│   └── amzn_stock.csv                 # AMZN OHLCV hisse verisi (2010-2025)
├── notebooks/
│   ├── 01_data_exploration.ipynb      # [Eski Deney] İlk EDA ve 80/20 train/test
│   ├── 02_model_experiments.ipynb     # [Eski Deney] Raw Close HPO ve hatalı GRU (RMSE $10.09)
│   ├── 03_time_series_analysis.ipynb  # [Aktif] Stationarity (ADF/KPSS), ACF/PACF ve Baseline analizi
│   └── 04_log_return_final_experiment.ipynb # [Final] Log Return, DL, Early Stopping ve Karşılaştırma
├── results/
│   ├── final_model_comparison.csv     # Final metrik tablosu (Log Return)
│   ├── final_price_comparison.csv     # 4+ basamak hassasiyetli Price RMSE sonuçları
│   ├── learning_curves.png            # LSTM & GRU Train/Val loss öğrenme eğrileri
│   └── final_prediction_comparison.png# Test seti gerçek vs tahmin fiyat grafiği
├── src/
│   ├── data_utils.py                  # Log return, 70/15/15 split ve sequence üretimi
│   ├── models.py                      # PyTorch LSTMModel ve GRUModel mimarileri
│   ├── train.py                       # DataLoader, loss tracking ve deepcopy early stopping
│   ├── evaluate.py                    # RMSE, MAE, Directional Acc ve Price reconstruction
│   └── main.py                        # Tek komutla çalışan bağımsız final pipeline
├── requirements.txt                   # Gerekli kütüphaneler
└── README.md                          # Proje dokümantasyonu
```

---

## 🚀 Kurulum ve Çalıştırma

### 1. Ortamın Hazırlanması

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Final Pipeline'ın Çalıştırılması

Tüm baseline'ları, model eğitimlerini, test değerlendirmelerini ve grafik üretimini tek komutla çalıştırmak için:

```bash
python3 src/main.py
```

### 3. Notebook İncelemesi

Detaylı görselleştirmeler, adım adım istatistiksel açıklamalar ve interaktif grafikler için `notebooks/04_log_return_final_experiment.ipynb` dosyasını çalıştırabilirsiniz.