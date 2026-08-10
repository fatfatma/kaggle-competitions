# Kaggle NLP - Disaster Tweets

[NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started) 

**Task:**  Classifying whether a tweet describes a real disaster or is a metaphorical expression (binary classification).

F1 Score

---

## Project

```
kaggle_nlp/
├── codes/              # Model kodları
├── data/                # train.csv, test.csv, sample_submission.csv
├── submissions/         # Kaggle'a yüklenen submission dosyaları
├── README.md
└── requirements.txt
```

---

## Denenen Yaklaşımlar ve Skorlar

| # | approach | CV F1 | Kaggle score | Note |
|---|----------|-------|--------------|-----|
| 1 | TF-IDF + Logistic Regression (baseline, `C=1.0`) | 0.7414 | 0.79865 | first baseline |
| 2 | TF-IDF + Logistic Regression (GridSearchCV optimize) | 0.7497 | 0.80140 | `C=5.0`, `max_features=20000`, `ngram_range=(1,2)`, `min_df=1` |
| 3 | **DistilBERT (fine-tuned, 2 epoch)** | **0.8165** | | Colab GPU (T4) |


---

## Denenen Modeller (Karşılaştırma)

5-fold Stratified CV (shuffled)

| Model | CV F1 |
|-------|-------|
| Logistic Regression | 0.7414 |
| LinearSVC | 0.7337 |
| Multinomial Naive Bayes | 0.7325 |

---

## Kullanılan Yöntem Özeti

1. **Data cleaning:** Aggressive cleaning for classic ML (URL, mention, hashtag, punctuation removed); 1. Minimal cleanup for BERT (only URL deleted, the rest of the context left to BERT)
2. **Baseline:** TF-IDF (unigram+bigram) + Logistic Regression
3. **Error analysis:** False positive/negative examples were examined — the model was most often wrong in metaphorical expressions (e.g., "slicker than an oil spill")
4. **Cross-validation:** `StratifiedKFold(shuffle=True)` was used — since the data was sorted by the `keyword` column, `shuffle=False` gave misleadingly low scores
5. **Model comparison:** Compared with Logistic Regression, LinearSVC, Multinomial Naive Bayes CV
6. **Hyperparameter optimization:** `Pipeline` + `GridSearchCV` were used to optimize TF-IDF and LogReg parameters together
7. **BERT fine-tuning:** `distilbert-base-uncased`, Colab T4 GPU, 2 epoch, `learning_rate=2e-5`, `max_length=64`
---

## setup

```bash
pip install -r requirements.txt
```

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## Sıradaki Adımlar

