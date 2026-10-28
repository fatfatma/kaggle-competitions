from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'

train_encoded = pd.read_csv(PROCESSED_DIR / 'train_processed.csv')
test_encoded = pd.read_csv(PROCESSED_DIR / 'test_processed.csv')

print(train_encoded.shape)
print(test_encoded.shape)

# ---- Feature Engineering ----
new_features_train = pd.DataFrame({
    'TotalSF': train_encoded['1stFlrSF'] + train_encoded['2ndFlrSF'] + train_encoded['TotalBsmtSF'],
    'TotalBath': train_encoded['FullBath'] + train_encoded['BsmtFullBath'] + 0.5 * (train_encoded['HalfBath'] + train_encoded['BsmtHalfBath']),
    'HouseAge': train_encoded['YrSold'] - train_encoded['YearBuilt'],
    'IsRemodeled': (train_encoded['YearBuilt'] != train_encoded['YearRemodAdd']).astype(int)
})
train_encoded = pd.concat([train_encoded, new_features_train], axis=1)

new_features_test = pd.DataFrame({
    'TotalSF': test_encoded['1stFlrSF'] + test_encoded['2ndFlrSF'] + test_encoded['TotalBsmtSF'],
    'TotalBath': test_encoded['FullBath'] + test_encoded['BsmtFullBath'] + 0.5 * (test_encoded['HalfBath'] + test_encoded['BsmtHalfBath']),
    'HouseAge': test_encoded['YrSold'] - test_encoded['YearBuilt'],
    'IsRemodeled': (test_encoded['YearBuilt'] != test_encoded['YearRemodAdd']).astype(int)
})
test_encoded = pd.concat([test_encoded, new_features_test], axis=1)

# Artık gereksiz hale gelen ham sütunları çıkar
cols_to_drop = ['1stFlrSF', '2ndFlrSF', 'TotalBsmtSF',
                 'FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath',
                 'YearBuilt', 'YrSold', 'YearRemodAdd']

train_encoded = train_encoded.drop(columns=cols_to_drop)
test_encoded = test_encoded.drop(columns=cols_to_drop)

print(train_encoded.shape)
print(test_encoded.shape)

# ---- X / y ayrımı ----
X = train_encoded.drop(columns=['SalePrice', 'SalePrice_log', 'Id'])
y = train_encoded['SalePrice_log']

print(X.shape)
print(y.shape)

# ---- Train / Validation split ----
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- Model ----
gb_model_fe = GradientBoostingRegressor(
    n_estimators=500,
    max_depth=3,
    learning_rate=0.02,
    random_state=42
)
gb_model_fe.fit(X_train, y_train)

y_pred_train_fe = gb_model_fe.predict(X_train)
y_pred_val_fe = gb_model_fe.predict(X_val)

rmse_train_fe = np.sqrt(mean_squared_error(y_train, y_pred_train_fe))
rmse_val_fe = np.sqrt(mean_squared_error(y_val, y_pred_val_fe))

print("GB (Feature Engineered) - TRAIN RMSE:", rmse_train_fe)
print("GB (Feature Engineered) - VALIDATION RMSE:", rmse_val_fe)

# En iyi modeli TÜM train verisiyle yeniden eğit (artık validation ayırmıyoruz)
final_model_fe = GradientBoostingRegressor(
    n_estimators=500,
    max_depth=3,
    learning_rate=0.02,
    random_state=42
)
final_model_fe.fit(X, y)

# Test verisi üzerinde tahmin yap
test_ids = test_encoded['Id']
X_test_final = test_encoded.drop(columns=['Id'])

test_predictions_log = final_model_fe.predict(X_test_final)

# Log ölçeğinden gerçek dolar ölçeğine geri çevir
test_predictions = np.expm1(test_predictions_log)

# Kaggle formatında submission dosyası oluştur
submission_fe = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': test_predictions
})

print(submission_fe.head())
print(submission_fe.shape)

# Kaydet
SUBMISSIONS_DIR = BASE_DIR / 'submissions'
SUBMISSIONS_DIR.mkdir(exist_ok=True)

submission_fe.to_csv(SUBMISSIONS_DIR / 'submission_v2_gb_fe.csv', index=False)
print("Kaydedildi:", SUBMISSIONS_DIR / 'submission_v2_gb_fe.csv')