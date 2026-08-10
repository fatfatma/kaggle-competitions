"""
NLP with Disaster Tweets - Baseline Model
 TF-IDF + Logistic Regression 
"""

import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import f1_score, classification_report

# 1. LOAD DATA
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")


print(f"Train boyutu: {train.shape}")
print(f"Test boyutu: {test.shape}")
print(f"\nHedef değişken dağılımı:\n{train['target'].value_counts()}")


# 2. CLEANING & PREPROCESSING
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)   # URL
    text = re.sub(r"@\w+", " ", text)                     # mention
    text = re.sub(r"#", " ", text)                        # hashtag 
    text = re.sub(r"[^a-zA-Z\s]", " ", text)               # punctiation and numbers
    text = re.sub(r"\s+", " ", text).strip()
    return text


train["clean_text"] = train["text"].apply(clean_text)
test["clean_text"] = test["text"].apply(clean_text)


# 3. Train / Validation 

X_train, X_val, y_train, y_val = train_test_split(
    train["clean_text"], train["target"],
    test_size=0.2, random_state=42, stratify=train["target"]
)


# 4. TF-IDF vectorization

vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),   # unigram + bigram
    min_df=2,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

# 5. Model eğitimi

model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
model.fit(X_train_tfidf, y_train)

val_preds = model.predict(X_val_tfidf)
val_f1 = f1_score(y_val, val_preds)

print(f"\nValidation F1 skoru: {val_f1:.4f}")
print("\nDetaylı rapor:")
print(classification_report(y_val, val_preds))

# 6b. Error Analysis: What is the model predicting incorrectly?

# we kept the original text for error analysis, not the cleaned version

val_df = pd.DataFrame({
    "text": train.loc[X_val.index, "text"],   # orijinal, temizlenmemiş metin
    "true_label": y_val,
    "pred_label": val_preds
})


#filter the errors
errors = val_df[val_df["true_label"] != val_df["pred_label"]]
print(f"\nToplam {len(errors)} / {len(val_df)} yanlış tahmin var")

# False Negative
false_negatives = errors[(errors["true_label"] == 1) & (errors["pred_label"] == 0)]
print(f"\n--- FALSE NEGATIVE örnekleri (kaçırılan afetler) ---")
print(false_negatives["text"].head(10).to_string(index=False))

# False Positive
false_positives = errors[(errors["true_label"] == 0) & (errors["pred_label"] == 1)]
print(f"\n--- FALSE POSITIVE örnekleri (yanlış alarm) ---")
print(false_positives["text"].head(10).to_string(index=False))



# 6. AGAİN TRAİN USİNG ALL TRAİN DATA (MUCH STRONGER MODEL) AND PREDİCT ON TEST DATA

X_full_tfidf = vectorizer.fit_transform(train["clean_text"])
X_test_tfidf = vectorizer.transform(test["clean_text"])

final_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
final_model.fit(X_full_tfidf, train["target"])

test_preds = final_model.predict(X_test_tfidf)


# 7. Submission 

submission = pd.DataFrame({
    "id": test["id"],
    "target": test_preds
})
submission.to_csv("submission.csv", index=False)

print("\nsubmission.csv is ready.")
print(submission.head())