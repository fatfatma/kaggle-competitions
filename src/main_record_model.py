from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


# 1. Orijinal Verileri Yukleme
train_df = pd.read_csv(DATA_DIR / "train.csv", sep=";", encoding="cp1252")
test_df = pd.read_csv(DATA_DIR / "test.csv")

train_df["Age"] = pd.to_numeric(train_df["Age"], errors="coerce")
test_df["Age"] = pd.to_numeric(test_df["Age"], errors="coerce")
train_df["Fare"] = pd.to_numeric(train_df["Fare"], errors="coerce")
test_df["Fare"] = pd.to_numeric(test_df["Fare"], errors="coerce")

# Birleştirip Grup Özellikleri Çıkarma
combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
train_target = combined.iloc[: len(train_df)]["Survived"].astype(int).reset_index(drop=True)


def add_oof_target_encoding(train_part, test_part, column_name, target, n_splits=5, smoothing=10, random_state=42):
    """Leakage-free target encoding for a categorical/group column."""
    global_mean = target.mean()
    oof_encoded = pd.Series(index=train_part.index, dtype=float)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, valid_idx in skf.split(train_part, target):
        fold_train = train_part.iloc[train_idx]
        fold_target = target.iloc[train_idx]

        stats = fold_train.groupby(column_name).apply(
            lambda x: fold_target.loc[x.index].agg(["mean", "count"])
        )

        if isinstance(stats, pd.Series):
            stats = stats.to_frame().T
            stats.index.name = column_name
            stats.columns = ["mean", "count"]

        smoothing_values = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
        fold_map = smoothing_values.to_dict()

        oof_encoded.iloc[valid_idx] = train_part.iloc[valid_idx][column_name].map(fold_map).fillna(global_mean)

    full_stats = train_part.groupby(column_name).apply(
        lambda x: target.loc[x.index].agg(["mean", "count"])
    )
    if isinstance(full_stats, pd.Series):
        full_stats = full_stats.to_frame().T
        full_stats.index.name = column_name
        full_stats.columns = ["mean", "count"]

    full_smoothing = (full_stats["count"] * full_stats["mean"] + smoothing * global_mean) / (full_stats["count"] + smoothing)
    test_encoded = test_part[column_name].map(full_smoothing.to_dict()).fillna(global_mean)

    return oof_encoded, test_encoded


# Soyadi cikarma
combined["Surname"] = combined["Name"].apply(lambda x: x.split(",")[0].strip())
combined["Name_Length"] = combined["Name"].apply(len)

# Bilet oneki
combined["Ticket_Prefix"] = combined["Ticket"].apply(
    lambda x: x.split()[0].replace(".", "").replace("/", "") if not str(x).isdigit() else "NUM"
)

# Bilet ve soyadi bazli grup buyuklugu
combined["Ticket_Frequency"] = combined.groupby("Ticket")["Ticket"].transform("count")
combined["Surname_Frequency"] = combined.groupby("Surname")["Surname"].transform("count")

# Unvan
combined["Title"] = combined["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
combined["Title"] = combined["Title"].replace([
    "Lady", "Countess", "Capt", "Col", "Don",
    "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona",
], "Rare")
combined["Title"] = combined["Title"].replace(["Mlle", "Ms"], "Miss")
combined["Title"] = combined["Title"].replace("Mme", "Mrs")

# Yas ve ucret doldurma
combined["Age"] = combined.groupby("Title")["Age"].transform(lambda x: x.fillna(x.median()))
combined["Fare"] = combined["Fare"].fillna(combined["Fare"].median())
combined["Cabin"] = combined["Cabin"].fillna("U")
combined["Deck"] = combined["Cabin"].str[0]
combined["Embarked"] = combined["Embarked"].fillna("S")

# Aile ve kisi basi ucret
combined["FamilySize"] = combined["SibSp"] + combined["Parch"] + 1
combined["IsAlone"] = (combined["FamilySize"] == 1).astype(int)
combined["Fare_Per_Person"] = combined["Fare"] / combined["FamilySize"]
combined["Fare_Log"] = np.log1p(combined["Fare"])
combined["Real_Fare"] = combined["Fare"] / combined["Ticket_Frequency"]

# Bilet grubu survival rate
combined["Is_Woman_Or_Child"] = ((combined["Title"].isin(["Miss", "Mrs", "Master"])) | (combined["Age"] < 14)).astype(int)
combined["Ticket_Woman_Child_Count"] = combined.groupby("Ticket")["Is_Woman_Or_Child"].transform("sum")

# OOF target encoding
train_part = combined.iloc[: len(train_df)].copy()
test_part = combined.iloc[len(train_df):].copy()

ticket_te_train, ticket_te_test = add_oof_target_encoding(train_part, test_part, "Ticket", train_target)
surname_te_train, surname_te_test = add_oof_target_encoding(train_part, test_part, "Surname", train_target)

combined.loc[: len(train_df) - 1, "Ticket_Group_Survival_TE"] = ticket_te_train.values
combined.loc[len(train_df):, "Ticket_Group_Survival_TE"] = ticket_te_test.values
combined.loc[: len(train_df) - 1, "Surname_Group_Survival_TE"] = surname_te_train.values
combined.loc[len(train_df):, "Surname_Group_Survival_TE"] = surname_te_test.values

# Split train / test
train_data = combined.iloc[: len(train_df)].copy()
test_data = combined.iloc[len(train_df):].copy()

# One-Hot Encoding
cat_cols = ["Sex", "Embarked", "Title", "Deck", "Pclass", "Ticket_Prefix"]
combined_encoded = pd.get_dummies(combined, columns=cat_cols, drop_first=True)

drop_cols = ["PassengerId", "Name", "Ticket", "Cabin", "Surname", "Survived"]
features = [c for c in combined_encoded.columns if c not in drop_cols]

X_train = combined_encoded.iloc[: len(train_df)][features].copy()
y_train = train_data["Survived"].astype(int)
X_test = combined_encoded.iloc[len(train_df):][features].copy()

# Model ve threshold optimizasyonu
xgb = XGBClassifier(
    n_estimators=350,
    max_depth=4,
    learning_rate=0.015,
    subsample=0.85,
    colsample_bytree=0.6,
    random_state=42,
    eval_metric="logloss",
)

xgb.fit(X_train, y_train)
train_probs = xgb.predict_proba(X_train)[:, 1]

best_thresh = 0.5
best_acc = 0.0
for thresh in np.arange(0.30, 0.70, 0.01):
    preds = (train_probs >= thresh).astype(int)
    acc = accuracy_score(y_train, preds)
    if acc > best_acc:
        best_acc = acc
        best_thresh = thresh

print(f"Optimum karar eşiği: {best_thresh:.2f}")
print(f"Bu eşikle train accuracy: %{best_acc * 100:.2f}")

test_probs = xgb.predict_proba(X_test)[:, 1]
final_preds = (test_probs >= best_thresh).astype(int)

submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"].astype(int),
    "Survived": final_preds,
})

submission_path = BASE_DIR / "submission_max_threshold.csv"
submission.to_csv(submission_path, index=False)
print("submission_max_threshold.csv başarıyla oluşturuldu!")
