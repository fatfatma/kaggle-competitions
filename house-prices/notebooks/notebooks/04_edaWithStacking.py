from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'

train_encoded = pd.read_csv(PROCESSED_DIR / 'train_processed.csv')
test_encoded = pd.read_csv(PROCESSED_DIR / 'test_processed.csv')

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

cols_to_drop = ['1stFlrSF', '2ndFlrSF', 'TotalBsmtSF',
                 'FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath',
                 'YearBuilt', 'YrSold', 'YearRemodAdd']
train_encoded = train_encoded.drop(columns=cols_to_drop)
test_encoded = test_encoded.drop(columns=cols_to_drop)

X = train_encoded.drop(columns=['SalePrice', 'SalePrice_log', 'Id'])
y = train_encoded['SalePrice_log']

print(X.shape)
print(y.shape)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# ---- Best Parameters using Optuna----
ridge_params = {'alpha': 19.389606583784072}

gb_params = {
    'n_estimators': 614,
    'learning_rate': 0.044512337856220024,
    'max_depth': 3,
    'min_samples_leaf': 4,
    'subsample': 0.6437890726267242
}

xgb_params = {
    'n_estimators': 744,
    'learning_rate': 0.016996151559754598,
    'max_depth': 4,
    'subsample': 0.5329832023858139,
    'colsample_bytree': 0.7032128098604563,
    'reg_alpha': 0.04523683336518137,
    'reg_lambda': 0.01632544051497689
}

# ---- Generate prediction OOF (Out-of-Fold)  ----
oof_ridge = np.zeros(len(X))
oof_gb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))

for train_idx, val_idx in kf.split(X):
    X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

    r = Ridge(**ridge_params)
    r.fit(X_tr, y_tr)
    oof_ridge[val_idx] = r.predict(X_va)

    g = GradientBoostingRegressor(**gb_params, random_state=42)
    g.fit(X_tr, y_tr)
    oof_gb[val_idx] = g.predict(X_va)

    x = XGBRegressor(**xgb_params, random_state=42)
    x.fit(X_tr, y_tr)
    oof_xgb[val_idx] = x.predict(X_va)

print("OOF tahminleri oluşturuldu.")
print("Ridge OOF RMSE:", np.sqrt(mean_squared_error(y, oof_ridge)))
print("GB OOF RMSE:", np.sqrt(mean_squared_error(y, oof_gb)))
print("XGB OOF RMSE:", np.sqrt(mean_squared_error(y, oof_xgb)))


X_meta = pd.DataFrame({
    'ridge': oof_ridge,
    'gb': oof_gb,
    'xgb': oof_xgb
})

from sklearn.model_selection import cross_val_score

meta_model = LinearRegression()
scores = cross_val_score(meta_model, X_meta, y, cv=kf, scoring='neg_root_mean_squared_error')
meta_rmse = -scores.mean()
print("Meta-model (stacking) - CV RMSE:", meta_rmse)


meta_model.fit(X_meta, y)
print("Meta-model katsayıları:", dict(zip(X_meta.columns, meta_model.coef_)))
print("Meta-model intercept:", meta_model.intercept_)


ridge_final = Ridge(**ridge_params)
ridge_final.fit(X, y)

gb_final = GradientBoostingRegressor(**gb_params, random_state=42)
gb_final.fit(X, y)

xgb_final = XGBRegressor(**xgb_params, random_state=42)
xgb_final.fit(X, y)

X_test_final = test_encoded.drop(columns=['Id'])
test_ids = test_encoded['Id']

pred_ridge_test = ridge_final.predict(X_test_final)
pred_gb_test = gb_final.predict(X_test_final)
pred_xgb_test = xgb_final.predict(X_test_final)

X_meta_test = pd.DataFrame({
    'ridge': pred_ridge_test,
    'gb': pred_gb_test,
    'xgb': pred_xgb_test
})

final_pred_log = meta_model.predict(X_meta_test)
final_pred = np.expm1(final_pred_log)

submission = pd.DataFrame({'Id': test_ids, 'SalePrice': final_pred})

SUBMISSIONS_DIR = BASE_DIR / 'submissions'
SUBMISSIONS_DIR.mkdir(exist_ok=True)
submission.to_csv(SUBMISSIONS_DIR / 'submission_v4_stacking.csv', index=False)

print(submission.head())
print(submission.shape)

