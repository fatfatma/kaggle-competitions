"""
Store Sales - Time Series Forecasting
Baseline Pipeline: Veri Birleştirme -> Feature Engineering -> Model -> Submission

Bu script train.csv, test.csv, stores.csv, oil.csv, holidays_events.csv
dosyalarını birleştirir, zaman bazlı feature'lar üretir ve bir LightGBM
modeliyle ilk (baseline) submission.csv dosyasını oluşturur.

Kurulum:
    pip install lightgbm

Çalıştırma:
    python generate_submission.py
"""

import numpy as np
import pandas as pd
import lightgbm as lgb

DATA_DIR = "data"
OUTPUT_PATH = "submission.csv"

CATEGORICAL_COLS = ["family", "store_nbr", "city", "state", "store_type", "cluster"]


# ---------------------------------------------------------------------------
# 1. Veri yükleme
# ---------------------------------------------------------------------------

def load_data():
    train = pd.read_csv(f"{DATA_DIR}/train.csv", parse_dates=["date"])
    test = pd.read_csv(f"{DATA_DIR}/test.csv", parse_dates=["date"])
    stores = pd.read_csv(f"{DATA_DIR}/stores.csv")
    oil = pd.read_csv(f"{DATA_DIR}/oil.csv", parse_dates=["date"])
    holidays = pd.read_csv(f"{DATA_DIR}/holidays_events.csv", parse_dates=["date"])
    return train, test, stores, oil, holidays


# ---------------------------------------------------------------------------
# 2. Holiday temizleme (Adım 5'te çıkardığımız kurallar)
# ---------------------------------------------------------------------------

def build_holiday_tables(holidays: pd.DataFrame):
    """
    Ham holidays_events verisini üç ayrı, kullanıma hazır tabloya çevirir:
      - national_holidays: sadece 'date'
      - regional_holidays: 'date' + 'state'
      - local_holidays:    'date' + 'city'
      - work_days:         tatil sayılmayan, telafi çalışma günleri (date)
      - events:            resmi tatil olmayan ama satışı etkileyebilecek günler (date)
    """
    is_actual_holiday = (
        ((holidays["type"] == "Holiday") & (holidays["transferred"] == False))
        | (holidays["type"] == "Transfer")
        | (holidays["type"] == "Additional")
        | (holidays["type"] == "Bridge")
    )
    real_holidays = holidays[is_actual_holiday].copy()

    national_holidays = (
        real_holidays[real_holidays["locale"] == "National"][["date"]]
        .drop_duplicates()
        .assign(is_holiday_national=1)
    )

    regional_holidays = (
        real_holidays[real_holidays["locale"] == "Regional"][["date", "locale_name"]]
        .rename(columns={"locale_name": "state"})
        .drop_duplicates()
        .assign(is_holiday_regional=1)
    )

    local_holidays = (
        real_holidays[real_holidays["locale"] == "Local"][["date", "locale_name"]]
        .rename(columns={"locale_name": "city"})
        .drop_duplicates()
        .assign(is_holiday_local=1)
    )

    work_days = (
        holidays[holidays["type"] == "Work Day"][["date"]]
        .drop_duplicates()
        .assign(is_work_day=1)
    )

    events = (
        holidays[holidays["type"] == "Event"][["date"]]
        .drop_duplicates()
        .assign(is_event=1)
    )

    return national_holidays, regional_holidays, local_holidays, work_days, events


def add_holiday_features(df: pd.DataFrame, stores: pd.DataFrame, holiday_tables) -> pd.DataFrame:
    """df'e (train veya test) mağaza konumuna göre doğru tatil bilgisini ekler."""
    national_holidays, regional_holidays, local_holidays, work_days, events = holiday_tables

    df = df.merge(stores[["store_nbr", "city", "state", "type", "cluster"]], on="store_nbr", how="left")
    df = df.rename(columns={"type": "store_type"})

    df = df.merge(national_holidays, on="date", how="left")
    df = df.merge(regional_holidays, on=["date", "state"], how="left")
    df = df.merge(local_holidays, on=["date", "city"], how="left")
    df = df.merge(work_days, on="date", how="left")
    df = df.merge(events, on="date", how="left")

    for col in ["is_holiday_national", "is_holiday_regional", "is_holiday_local", "is_work_day", "is_event"]:
        df[col] = df[col].fillna(0).astype(int)

    # Bir gün herhangi bir seviyede (national/regional/local) tatilse VE work_day ile
    # telafi edilmemişse "gerçek tatil" sayılır.
    df["is_holiday"] = (
        (df["is_holiday_national"] | df["is_holiday_regional"] | df["is_holiday_local"])
        & (df["is_work_day"] == 0)
    ).astype(int)

    df = df.drop(columns=["is_holiday_national", "is_holiday_regional", "is_holiday_local", "is_work_day"])
    return df


# ---------------------------------------------------------------------------
# 3. Oil feature'ı (hafta sonu / tatil boşluklarını doldurma)
# ---------------------------------------------------------------------------

def add_oil_feature(df: pd.DataFrame, oil: pd.DataFrame) -> pd.DataFrame:
    full_dates = pd.DataFrame({"date": pd.date_range(oil["date"].min(), oil["date"].max())})
    oil_full = full_dates.merge(oil, on="date", how="left")
    oil_full["dcoilwtico"] = oil_full["dcoilwtico"].ffill().bfill()
    oil_full = oil_full.rename(columns={"dcoilwtico": "oil_price"})

    df = df.merge(oil_full, on="date", how="left")
    df["oil_price"] = df["oil_price"].ffill().bfill()
    return df


# ---------------------------------------------------------------------------
# 4. Tarih feature'ları
# ---------------------------------------------------------------------------

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df["dayofyear"] = df["date"].dt.dayofyear
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # Kamu çalışanlarının maaş aldığı günler (ayın 15'i ve son günü) -> satışı etkileyebilir
    df["is_wage_day"] = (df["day"] == 15) | (df["date"].dt.is_month_end)
    df["is_wage_day"] = df["is_wage_day"].astype(int)

    return df


# ---------------------------------------------------------------------------
# 5. Tüm feature pipeline'ını birleştir
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame, stores, oil, holiday_tables) -> pd.DataFrame:
    df = add_holiday_features(df, stores, holiday_tables)
    df = add_oil_feature(df, oil)
    df = add_date_features(df)

    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")

    return df


# ---------------------------------------------------------------------------
# 6. Model eğitimi (zaman bazlı train/validation split)
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "store_nbr", "family", "onpromotion",
    "city", "state", "store_type", "cluster",
    "is_holiday", "is_event", "oil_price",
    "year", "month", "day", "dayofweek", "dayofyear",
    "is_weekend", "is_wage_day",
]


def train_model(train_df: pd.DataFrame):
    train_df = train_df.copy()
    train_df["sales_log"] = np.log1p(train_df["sales"])

    # Zaman serisi olduğu için rastgele split YAPMIYORUZ.
    # Son 15 gün validation, geri kalanı train (test seti de son tarihten sonraki 15 gün).
    cutoff = train_df["date"].max() - pd.Timedelta(days=15)
    tr = train_df[train_df["date"] <= cutoff]
    val = train_df[train_df["date"] > cutoff]

    X_tr, y_tr = tr[FEATURE_COLS], tr["sales_log"]
    X_val, y_val = val[FEATURE_COLS], val["sales_log"]

    model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=128,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        categorical_feature=CATEGORICAL_COLS,
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
    )

    print("\nEn iyi iterasyon:", model.best_iteration_)
    print("Validation RMSE (log ölçeğinde):", model.best_score_["valid_0"]["rmse"])

    return model


# ---------------------------------------------------------------------------
# 7. Tahmin ve submission dosyası üretimi
# ---------------------------------------------------------------------------

def predict_and_save(model, test_df: pd.DataFrame):
    preds_log = model.predict(test_df[FEATURE_COLS])
    preds = np.expm1(preds_log)
    preds = np.clip(preds, 0, None)  # negatif satış olamaz

    submission = pd.DataFrame({
        "id": test_df["id"],
        "sales": preds,
    })
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSubmission dosyası kaydedildi: {OUTPUT_PATH}")
    print(submission.head())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    train, test, stores, oil, holidays = load_data()
    holiday_tables = build_holiday_tables(holidays)

    print("Feature'lar oluşturuluyor (train)...")
    train_feat = build_features(train, stores, oil, holiday_tables)

    print("Feature'lar oluşturuluyor (test)...")
    test_feat = build_features(test, stores, oil, holiday_tables)

    print("Model eğitiliyor...")
    model = train_model(train_feat)

    print("Tahmin yapılıyor ve submission.csv oluşturuluyor...")
    predict_and_save(model, test_feat)


if __name__ == "__main__":
    main()