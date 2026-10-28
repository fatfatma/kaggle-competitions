from pathlib import Path
from itertools import combinations
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

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

# ---- Best Parameters using Optuna (03'ten) ----
xgb_params = {
    'n_estimators': 744,
    'learning_rate': 0.016996151559754598,
    'max_depth': 4,
    'subsample': 0.5329832023858139,
    'colsample_bytree': 0.7032128098604563,
    'reg_alpha': 0.04523683336518137,
    'reg_lambda': 0.01632544051497689
}

# ---- XGBoost ile Feature Importance Analizi ----
xgb_for_importance = XGBRegressor(**xgb_params, random_state=42)
xgb_for_importance.fit(X, y)

importances = pd.Series(xgb_for_importance.feature_importances_, index=X.columns)
importances_sorted = importances.sort_values(ascending=False)

zero_importance_cols = importances_sorted[importances_sorted == 0].index.tolist()

# ---- Sıfır önemli sütunları çıkar (v5 ile aynı, 175 sütun) ----
X_reduced = X.drop(columns=zero_importance_cols)
test_encoded_reduced = test_encoded.drop(columns=zero_importance_cols)

print("Önceki sütun sayısı:", X.shape[1])
print("Sonraki sütun sayısı:", X_reduced.shape[1])

# ============================================================
# 5 MODELİN OPTUNA İLE BULUNAN EN İYİ PARAMETRELERİ (03'ten)
# ============================================================

ridge_params = {'alpha': 19.389606583784072}

gb_params = {
    'n_estimators': 614,
    'learning_rate': 0.044512337856220024,
    'max_depth': 3,
    'min_samples_leaf': 4,
    'subsample': 0.6437890726267242
}

rf_params = {
    'n_estimators': 69,
    'max_depth': 10,
    'min_samples_leaf': 2,
    'max_features': None
}

lgbm_params = {
    'n_estimators': 700,
    'learning_rate': 0.023087719924610256,
    'max_depth': 7,
    'num_leaves': 25,
    'subsample': 0.5323773871216173,
    'colsample_bytree': 0.5943965726983964,
    'reg_alpha': 0.003324614566556779,
    'reg_lambda': 0.0025855300706014296,
    'verbose': -1
}

# ============================================================
# 5 MODEL İÇİN OOF (OUT-OF-FOLD) TAHMİNLERİ
# ============================================================

oof_ridge = np.zeros(len(X_reduced))
oof_rf = np.zeros(len(X_reduced))
oof_gb = np.zeros(len(X_reduced))
oof_xgb = np.zeros(len(X_reduced))
oof_lgbm = np.zeros(len(X_reduced))

for fold_num, (train_idx, val_idx) in enumerate(kf.split(X_reduced), 1):
    X_tr, X_va = X_reduced.iloc[train_idx], X_reduced.iloc[val_idx]
    y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

    r = Ridge(**ridge_params)
    r.fit(X_tr, y_tr)
    oof_ridge[val_idx] = r.predict(X_va)

    rf = RandomForestRegressor(**rf_params, random_state=42)
    rf.fit(X_tr, y_tr)
    oof_rf[val_idx] = rf.predict(X_va)

    g = GradientBoostingRegressor(**gb_params, random_state=42)
    g.fit(X_tr, y_tr)
    oof_gb[val_idx] = g.predict(X_va)

    x = XGBRegressor(**xgb_params, random_state=42)
    x.fit(X_tr, y_tr)
    oof_xgb[val_idx] = x.predict(X_va)

    lgbm = LGBMRegressor(**lgbm_params, random_state=42)
    lgbm.fit(X_tr, y_tr)
    oof_lgbm[val_idx] = lgbm.predict(X_va)

    print(f"Fold {fold_num} tamamlandı.", flush=True)

print("\n---- Tekil Model OOF RMSE'leri ----")
print("Ridge:", np.sqrt(mean_squared_error(y, oof_ridge)))
print("Random Forest:", np.sqrt(mean_squared_error(y, oof_rf)))
print("Gradient Boosting:", np.sqrt(mean_squared_error(y, oof_gb)))
print("XGBoost:", np.sqrt(mean_squared_error(y, oof_xgb)))
print("LightGBM:", np.sqrt(mean_squared_error(y, oof_lgbm)))

# ============================================================
# TÜM MODEL ALT KÜMELERİYLE STACKING DENEMESİ
# ============================================================

oof_dict = {
    'ridge': oof_ridge,
    'rf': oof_rf,
    'gb': oof_gb,
    'xgb': oof_xgb,
    'lgbm': oof_lgbm
}

model_names = list(oof_dict.keys())

stacking_results = {}

# 2'li, 3'lü, 4'lü ve 5'li tüm kombinasyonları dene
for r_size in range(2, len(model_names) + 1):
    for combo in combinations(model_names, r_size):
        X_meta_combo = pd.DataFrame({name: oof_dict[name] for name in combo})
        meta_model = LinearRegression()
        scores = cross_val_score(meta_model, X_meta_combo, y, cv=kf, scoring='neg_root_mean_squared_error')
        rmse = -scores.mean()
        combo_name = ' + '.join(combo)
        stacking_results[combo_name] = rmse

# Tekil modelleri de tabloya ekle (referans için)
tekil_results = {
    'ridge (tekil)': np.sqrt(mean_squared_error(y, oof_ridge)),
    'rf (tekil)': np.sqrt(mean_squared_error(y, oof_rf)),
    'gb (tekil)': np.sqrt(mean_squared_error(y, oof_gb)),
    'xgb (tekil)': np.sqrt(mean_squared_error(y, oof_xgb)),
    'lgbm (tekil)': np.sqrt(mean_squared_error(y, oof_lgbm)),
}

all_results = {**tekil_results, **stacking_results}

results_df = pd.DataFrame(list(all_results.items()), columns=['Kombinasyon', 'RMSE'])
results_df = results_df.sort_values('RMSE').reset_index(drop=True)

print("\n" + "=" * 60)
print("TÜM KOMBİNASYONLARIN SONUÇLARI (küçükten büyüğe)")
print("=" * 60)
print(results_df.to_string())

best_combo_name = results_df.iloc[0]['Kombinasyon']
best_rmse = results_df.iloc[0]['RMSE']
print(f"\nEN İYİ SONUÇ: {best_combo_name} -> RMSE: {best_rmse:.6f}")

# ============================================================
# EN İYİ KOMBİNASYONLA FİNAL MODEL + SUBMISSION
# ============================================================

if ' + ' in best_combo_name:
    best_models = best_combo_name.split(' + ')
else:
    best_models = [best_combo_name.replace(' (tekil)', '')]

print("Kullanılacak modeller:", best_models)

X_test_final = test_encoded_reduced.drop(columns=['Id'])
test_ids = test_encoded_reduced['Id']

# Sadece gereken modelleri tüm veriyle (X_reduced, y) yeniden eğit
final_models = {}
test_preds = {}

if 'ridge' in best_models:
    m = Ridge(**ridge_params)
    m.fit(X_reduced, y)
    final_models['ridge'] = m
    test_preds['ridge'] = m.predict(X_test_final)

if 'rf' in best_models:
    m = RandomForestRegressor(**rf_params, random_state=42)
    m.fit(X_reduced, y)
    final_models['rf'] = m
    test_preds['rf'] = m.predict(X_test_final)

if 'gb' in best_models:
    m = GradientBoostingRegressor(**gb_params, random_state=42)
    m.fit(X_reduced, y)
    final_models['gb'] = m
    test_preds['gb'] = m.predict(X_test_final)

if 'xgb' in best_models:
    m = XGBRegressor(**xgb_params, random_state=42)
    m.fit(X_reduced, y)
    final_models['xgb'] = m
    test_preds['xgb'] = m.predict(X_test_final)

if 'lgbm' in best_models:
    m = LGBMRegressor(**lgbm_params, random_state=42)
    m.fit(X_reduced, y)
    final_models['lgbm'] = m
    test_preds['lgbm'] = m.predict(X_test_final)

if len(best_models) > 1:
    # Stacking: meta-modeli en iyi kombinasyonun OOF tahminleriyle eğit
    X_meta_best = pd.DataFrame({name: oof_dict[name] for name in best_models})
    meta_model_final = LinearRegression()
    meta_model_final.fit(X_meta_best, y)

    X_meta_test = pd.DataFrame({name: test_preds[name] for name in best_models})
    final_pred_log = meta_model_final.predict(X_meta_test)

    print("Meta-model katsayıları:", dict(zip(best_models, meta_model_final.coef_)))
else:
    # Tek model kazandıysa, direkt onun tahminini kullan
    final_pred_log = test_preds[best_models[0]]

final_pred = np.expm1(final_pred_log)

submission = pd.DataFrame({'Id': test_ids, 'SalePrice': final_pred})

SUBMISSIONS_DIR = BASE_DIR / 'submissions'
SUBMISSIONS_DIR.mkdir(exist_ok=True)
submission.to_csv(SUBMISSIONS_DIR / 'submission_v7_full_stacking.csv', index=False)

print("\nKaydedildi:", SUBMISSIONS_DIR / 'submission_v7_full_stacking.csv')
print(submission.head())
print(submission.shape)
