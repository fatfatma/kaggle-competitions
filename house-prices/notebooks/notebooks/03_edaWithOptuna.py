from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import optuna

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'

train_encoded = pd.read_csv(PROCESSED_DIR / 'train_processed.csv')
test_encoded = pd.read_csv(PROCESSED_DIR / 'test_processed.csv')

# ---- Feature Engineering (02'den aynı mantık) ----
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

kf = KFold(n_splits=5, shuffle=True, random_state=42)  # tüm modellerde aynı fold'ları kullanmak için


#RİDGE 

def objective_ridge(trial):
    alpha = trial.suggest_float('alpha', 0.01, 100, log=True)
    
    model = Ridge(alpha=alpha, random_state=42)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring='neg_root_mean_squared_error')
    
    rmse = -scores.mean()
    return rmse

study_ridge = optuna.create_study(direction='minimize')
study_ridge.optimize(objective_ridge, n_trials=50)

print("Ridge - En iyi parametreler:", study_ridge.best_params)
print("Ridge - En iyi RMSE:", study_ridge.best_value)

#Random Forest


def objective_rf(trial):
    n_estimators = trial.suggest_int('n_estimators', 10, 100)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42
    )
    scores = cross_val_score(model, X, y, cv=kf, scoring='neg_root_mean_squared_error')
    rmse = -scores.mean()
    return rmse

study_rf = optuna.create_study(direction='minimize')
study_rf.optimize(objective_rf, n_trials=50)

print("Random Forest - En iyi parametreler:", study_rf.best_params)
print("Random Forest - En iyi RMSE:", study_rf.best_value)

def objective_gb(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 800)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    max_depth = trial.suggest_int('max_depth', 2, 6)
    min_samples_leaf= trial.suggest_int('min_samples_leaf', 1, 10)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    
    scores = cross_val_score(model, X, y, cv=kf, scoring='neg_root_mean_squared_error')
    rmse = -scores.mean()
    return rmse

study_gb = optuna.create_study(direction='minimize')
study_gb.optimize(objective_gb, n_trials=50)

print("Gradient Boosting - En iyi parametreler:", study_gb.best_params)
print("Gradient Boosting - En iyi RMSE:", study_gb.best_value)


#XGBoots 

from xgboost import XGBRegressor

def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 800),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 8),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
        'random_state': 42
    }
    model = XGBRegressor(**params)
    scores = cross_val_score(model, X, y, cv=kf, scoring='neg_root_mean_squared_error')
    return -scores.mean()

study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(objective_xgb, n_trials=40)

print("XGBoost - En iyi parametreler:", study_xgb.best_params)
print("XGBoost - En iyi RMSE:", study_xgb.best_value)


#LightBM

from lightgbm import LGBMRegressor

def objective_lgbm(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 800),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 8),
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
        'random_state': 42,
        'verbose': -1
    }
    model = LGBMRegressor(**params)
    scores = cross_val_score(model, X, y, cv=kf, scoring='neg_root_mean_squared_error')
    return -scores.mean()

study_lgbm = optuna.create_study(direction='minimize')
study_lgbm.optimize(objective_lgbm, n_trials=40)

print("LightGBM - En iyi parametreler:", study_lgbm.best_params)
print("LightGBM - En iyi RMSE:", study_lgbm.best_value)

# ============================================================
# FINAL MODELLER (Optuna ile bulunan en iyi parametrelerle)
# ============================================================

from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

ridge_final = Ridge(**study_ridge.best_params)
ridge_final.fit(X, y)

gb_final = GradientBoostingRegressor(
    **study_gb.best_params,
    random_state=42
)
gb_final.fit(X, y)

xgb_final = XGBRegressor(
    **study_xgb.best_params,
    random_state=42
)
xgb_final.fit(X, y)

print("Final modeller tüm veri üzerinde eğitildi.")

# ============================================================
# ENSEMBLE Cross Validation
# ============================================================

from sklearn.metrics import mean_squared_error

# Denenecek ağırlık kombinasyonları
weight_combos = {
    'equal':         (1/3, 1/3, 1/3),
    'xgb_heavy':      (0.2, 0.2, 0.6),
    'ridge_xgb_only': (0.5, 0.0, 0.5),
    'xgb_gb_only':    (0.0, 0.4, 0.6),
}

combo_scores = {name: [] for name in weight_combos}

for train_idx, val_idx in kf.split(X):

    X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

    # Ridge
    r = Ridge(**study_ridge.best_params)
    r.fit(X_tr, y_tr)
    pred_r = r.predict(X_va)

    # Gradient Boosting
    g = GradientBoostingRegressor(
        **study_gb.best_params,
        random_state=42
    )
    g.fit(X_tr, y_tr)
    pred_g = g.predict(X_va)

    # XGBoost
    x = XGBRegressor(
        **study_xgb.best_params,
        random_state=42
    )
    x.fit(X_tr, y_tr)
    pred_x = x.predict(X_va)

    for name, (w_r, w_g, w_x) in weight_combos.items():

        pred_combo = (
            w_r * pred_r +
            w_g * pred_g +
            w_x * pred_x
        )

        rmse_combo = np.sqrt(mean_squared_error(y_va, pred_combo))
        combo_scores[name].append(rmse_combo)

    
print("=" * 60)
print("TÜM SONUÇLARIN ÖZETİ")
print("=" * 60)

all_results = {
    'Ridge (tekil)': study_ridge.best_value,
    'Random Forest (tekil)': study_rf.best_value,
    'Gradient Boosting (tekil)': study_gb.best_value,
    'XGBoost (tekil)': study_xgb.best_value,
    'LightGBM (tekil)': study_lgbm.best_value,
}

for name, scores in combo_scores.items():
    all_results[f'Ensemble - {name}'] = np.mean(scores)

results_df = pd.DataFrame(
    list(all_results.items()),
    columns=['Model', 'RMSE']
)

results_df = results_df.sort_values(
    'RMSE'
).reset_index(drop=True)

print(results_df)

best_model_name = results_df.iloc[0]['Model']
best_rmse = results_df.iloc[0]['RMSE']

print(f"\nEN İYİ SONUÇ: {best_model_name} -> RMSE: {best_rmse:.6f}")

X_test_final = test_encoded.drop(columns=['Id'])
test_ids = test_encoded['Id']

pred_r_test = ridge_final.predict(X_test_final)
pred_g_test = gb_final.predict(X_test_final)
pred_x_test = xgb_final.predict(X_test_final)

if best_model_name.startswith('Ensemble'):
    combo_key = best_model_name.replace('Ensemble - ', '')
    w_r, w_g, w_x = weight_combos[combo_key]
    final_pred_log = (
        w_r * pred_r_test +
        w_g * pred_g_test +
        w_x * pred_x_test
    )

elif best_model_name == 'Ridge (tekil)':
    final_pred_log = pred_r_test

elif best_model_name == 'Gradient Boosting (tekil)':
    final_pred_log = pred_g_test

elif best_model_name == 'XGBoost (tekil)':
    final_pred_log = pred_x_test

else:
    print("UYARI: RF veya LightGBM kazandı.")
    final_pred_log = pred_x_test

final_pred = np.expm1(final_pred_log)

submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': final_pred
})

SUBMISSIONS_DIR = BASE_DIR / 'submissions'
SUBMISSIONS_DIR.mkdir(exist_ok=True)

submission.to_csv(
    SUBMISSIONS_DIR / 'submission_v4_optuna_ensemble.csv',
    index=False
)

print(submission.head())
print(submission.shape)