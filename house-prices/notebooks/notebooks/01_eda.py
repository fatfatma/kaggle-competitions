from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#first we will load the data and check the shape of the data and the first few rows of the data
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / 'data' / 'raw'

train = pd.read_csv(DATA_DIR / 'train.csv')
test = pd.read_csv(DATA_DIR / 'test.csv')
print(train.shape)
print(test.shape)

train.head()
train.info()

#OUTLIER ANALYSIS - GrLivArea çok büyük ama SalePrice düşük olan evler
#sağ altta iki nokta dikkat çekiyor onları bulmak için aşağıdaki kod
outliers = train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)]
print(outliers[['Id', 'GrLivArea', 'SalePrice']])

print("Çıkarmadan önce:", train.shape)

train = train[~((train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000))]

print("Çıkardıktan sonra:", train.shape)


# Our column list about Bodrum
bsmt_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']

# 1. Lines where Bsmt Exposure is missing but BsmtQual is filled.
exposure_missing = train[train['BsmtExposure'].isnull() & train['BsmtQual'].notnull()][bsmt_cols]

print(f"--- COUNT: {len(exposure_missing)} ---")
print(exposure_missing)
print("\n" + "="*50 + "\n")

# 2. Rows where BsmtFinType2 is missing but BsmtQual is populated.

finType2_missing = train[train['BsmtFinType2'].isnull() & train['BsmtQual'].notnull()][bsmt_cols]

print(f"--- COUNT2: {len(finType2_missing)} ---")
print(finType2_missing)

# For LotFrontage missing values (1 Group by Neighborhood and fill in missing values with the median LotFrontage of all the neighborhood)
print("...........................................................")

neighborhood_medians = train.groupby('Neighborhood')['LotFrontage'].median()

neighborhood_median_series = train.groupby('Neighborhood')['LotFrontage'].transform('median')
train['LotFrontage'] = train['LotFrontage'].fillna(neighborhood_median_series)

"""
#for control we will check the missing values in LotFrontage after filling the missing values with the median of the neighborhood
print(train['LotFrontage'].isnull().sum())
"""
# for filling test data

test['LotFrontage'] = test['LotFrontage'].fillna(test['Neighborhood'].map(neighborhood_medians))

#Group A "None" values in the basement columns to "None"
for col in ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtFinType1', 'MasVnrType']:
    train[col] = train[col].fillna('None')
    test[col] = test[col].fillna('None')


#Group B columns to "0"

for col in ['GarageYrBlt', 'MasVnrArea']:
    train[col] = train[col].fillna(0)
    test[col] = test[col].fillna(0)

#Group C values in the basement columns to "Mode"

electrical_mode = train['Electrical'].mode()[0]
train['Electrical'] = train['Electrical'].fillna(electrical_mode)
test['Electrical'] = test['Electrical'].fillna(electrical_mode)

for col in ['BsmtExposure', 'BsmtFinType2']:
    mode_val = train[col].mode()[0]
    train[col] = train[col].fillna(mode_val)
    test[col] = test[col].fillna(mode_val) 

# Kategorik gerçek eksikler -> mode
for col in ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'Functional', 'SaleType']:
    mode_val = train[col].mode()[0]
    test[col] = test[col].fillna(mode_val)

# Sayısal, "yok" anlamına gelen -> 0
for col in ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea']:
    test[col] = test[col].fillna(0)

######### Ordinal kategorik değişkenleri sayısala çevirme ###################
print(train['KitchenQual'].value_counts())

qual_map = {
    'None': 0,
    'Po': 1,
    'Fa': 2,
    'TA': 3,
    'Gd': 4,
    'Ex': 5
}


qual_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
             'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']

for col in qual_cols:
    train[col] = train[col].map(qual_map)
    test[col] = test[col].map(qual_map)


#BsmtExposure 
bsmt_exposure_map = {
    'None': 0,
    'No': 1,
    'Mn': 2,
    'Av': 3,
    'Gd': 4
}

train['BsmtExposure'] = train['BsmtExposure'].map(bsmt_exposure_map)
test['BsmtExposure'] = test['BsmtExposure'].map(bsmt_exposure_map)

#BsmtFinType1 and BsmtFinType2
bsmt_fin_type_map = {
    'None': 0,
    'Unf': 1,
    'LwQ': 2,
    'Rec': 3,
    'BLQ': 4,
    'ALQ': 5,
    'GLQ': 6
}           

train['BsmtFinType1'] = train['BsmtFinType1'].map(bsmt_fin_type_map)
train['BsmtFinType2'] = train['BsmtFinType2'].map(bsmt_fin_type_map)
test['BsmtFinType1'] = test['BsmtFinType1'].map(bsmt_fin_type_map)
test['BsmtFinType2'] = test['BsmtFinType2'].map(bsmt_fin_type_map)

# GarageFinish  

garage_finish_map = {
    'None': 0,
    'Unf': 1,
    'RFn': 2,
    'Fin': 3
}

train['GarageFinish'] = train['GarageFinish'].map(garage_finish_map)
test['GarageFinish'] = test['GarageFinish'].map(garage_finish_map)  

#LandSlope  
land_slope_map = {
    'Gtl': 0,
    'Mod': 1,
    'Sev': 2
}
train['LandSlope'] = train['LandSlope'].map(land_slope_map)
test['LandSlope'] = test['LandSlope'].map(land_slope_map)  

#PavedDrive
paved_drive_map = {
    'N': 0,
    'P': 1,
    'Y': 2
}   

train['PavedDrive'] = train['PavedDrive'].map(paved_drive_map)
test['PavedDrive'] = test['PavedDrive'].map(paved_drive_map)    

#FUnctional
functional_map = {
    'Sal': 0,
    'Sev': 1,
    'Maj2': 2, 
    'Maj1': 3,
    'Mod': 4,
    'Min2': 5,
    'Min1': 6,
    'Typ': 7
}   

train['Functional'] = train['Functional'].map(functional_map)
test['Functional'] = test['Functional'].map(functional_map)

'''
FOR CONTROL ALL COLUMS 
ordinal_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
                 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC',
                 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'GarageFinish',
                 'Functional', 'LandSlope', 'PavedDrive']

print(train[ordinal_cols].isnull().sum())
print(test[ordinal_cols].isnull().sum())

'''

# nominal kategorik değişkenleri sayısala çevirme one-hot


# ANALYSİS

categorical_cols = train.select_dtypes(include='object').columns.tolist()
print("Nominal sütun sayısı:", len(categorical_cols))
print(categorical_cols)

for col in categorical_cols:
    train_cats = set(train[col].unique())
    test_cats = set(test[col].unique())
    if train_cats != test_cats:
        print(f"{col}: train'de olup test'te olmayan -> {train_cats - test_cats}, test'te olup train'de olmayan -> {test_cats - train_cats}")

        
# SUMMARY: train'de olup test'te olmayan kategoriler var, ama test'te olup train'de olmayan hiç yok

print(".................................................................")

train_encoded = pd.get_dummies(train, columns=categorical_cols)
test_encoded = pd.get_dummies(test, columns=categorical_cols)

# test'i train'in sütunlarına hizala
test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)

print(train_encoded.shape)
print(test_encoded.shape)
print('SalePrice' in test_encoded.columns)

test_encoded = test_encoded.drop(columns=['SalePrice'])

# SALE PRİCE ANALYSIS AND LOG TRANSFORMATION
'''
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(train['SalePrice'], kde=True, ax=axes[0])
axes[0].set_title('SalePrice - Orijinal Dağılım')

sns.histplot(np.log1p(train['SalePrice']), kde=True, ax=axes[1])
axes[1].set_title('SalePrice - Log Dönüşümü Sonrası')

plt.tight_layout()
plt.show()

#BURADAKİ GRAFİKTEN YOLA ÇIKARAK NEDEN LOG DÖNÜŞÜMÜ YAPMAMIZ GEREKTİĞİNİ ANLADIK 
Çoğu regresyon modeli 
(özellikle Linear Regression gibi klasik yöntemler) 
hedef değişkenin normal dağılıma yakın olmasını sever — çarpık 
dağılımlarda büyük değerler (pahalı evler) modelin hata 
fonksiyonunu orantısız şekilde etkiler, model ucuz evlerdeki 
hatalardan çok pahalı evlerdeki hatalara odaklanmaya başlar. 
Log dönüşümü bu dengesizliği düzeltir.

'''
train_encoded['SalePrice_log'] = np.log1p(train_encoded['SalePrice'])

'''
DOĞRULAMAK İÇİN LOG DÖNÜŞÜMÜNÜ YAPTIĞIMIZI GÖRMEK İÇİN
print(train_encoded[['SalePrice', 'SalePrice_log']].head())
''' 
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
PROCESSED_DIR.mkdir(exist_ok=True)

train_encoded.to_csv(PROCESSED_DIR / 'train_processed.csv', index=False)
test_encoded.to_csv(PROCESSED_DIR / 'test_processed.csv', index=False)

print("Kaydedildi:", PROCESSED_DIR / 'train_processed.csv')
print("Kaydedildi:", PROCESSED_DIR / 'test_processed.csv')

#FEATURE ENGINEERING

X = train_encoded.drop(columns=['SalePrice', 'SalePrice_log', 'Id']) # Modele gereksiz girdi vermek istemiyoruz.

y = train_encoded['SalePrice_log']

print(X.shape)  
print(y.shape)

#Train-Validation 
'''
Training set (örneğin %80) → modeli bunun üzerinde eğitiyoruz
Validation set (örneğin %20) → model hiç görmediği bu veri üzerinde test ediyoruz, gerçek performansın tahminini alıyoruz

'''
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_val.shape)

#MODEL BASELİNE

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print("Validation RMSE (log ölçeğinde):", rmse)

from sklearn.metrics import mean_absolute_error

# Tahminleri ve gerçek değerleri orijinal dolar ölçeğine geri çevir
y_val_actual = np.expm1(y_val)
y_pred_actual = np.expm1(y_pred)

mae = mean_absolute_error(y_val_actual, y_pred_actual)
print("Validation MAE (gerçek dolar cinsinden):", mae)

print("Ortalama SalePrice:", train['SalePrice'].mean())
print("MAE / Ortalama SalePrice oranı (%):", (15154.95 / train['SalePrice'].mean()) * 100)

#RANDOM FOREST

from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_val)

rmse_rf = np.sqrt(mean_squared_error(y_val, y_pred_rf))
mae_rf = mean_absolute_error(np.expm1(y_val), np.expm1(y_pred_rf))

print("Random Forest - Validation RMSE (log):", rmse_rf)
print("Random Forest - Validation MAE ($):", mae_rf)

'''
# overfitting var mı testi 
y_pred_rf_train = rf_model.predict(X_train)
rmse_rf_train = np.sqrt(mean_squared_error(y_train, y_pred_rf_train))
print("Random Forest - TRAIN RMSE (log):", rmse_rf_train)
print("Random Forest - VALIDATION RMSE (log):", rmse_rf)

#OVERGİTTİNG VARMIŞ ONU ÇÖZECEĞİZ AŞAĞIDA

#Modelimiz doğru şeylere bakıyor, encoding stratejimiz sorunlu değilmiş. Yani hipotezimiz (One-Hot sütunları modeli şaşırtıyor) yanlış çıktı — bu da güzel bir şey, çünkü elimizdeki veri hazırlama sürecinin sağlam olduğunu doğruladı.
#Küçük veri setine sahip olduğumuz için olabilir 

rf_model_v2 = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=2,
    random_state=42
)
rf_model_v2.fit(X_train, y_train)

y_pred_train_v2 = rf_model_v2.predict(X_train)
y_pred_val_v2 = rf_model_v2.predict(X_val)

rmse_train_v2 = np.sqrt(mean_squared_error(y_train, y_pred_train_v2))
rmse_val_v2 = np.sqrt(mean_squared_error(y_val, y_pred_val_v2))

print("RF v2 - TRAIN RMSE:", rmse_train_v2)
print("RF v2 - VALIDATION RMSE:", rmse_val_v2)

print(".....................................................")
importances = pd.Series(rf_model_v2.feature_importances_, index=X_train.columns)
top_20 = importances.sort_values(ascending=False).head(20)
print(top_20)

'''

#GrandBoosting 

from sklearn.ensemble import GradientBoostingRegressor

gb_model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    random_state=42
)
gb_model.fit(X_train, y_train)

y_pred_train_gb = gb_model.predict(X_train)
y_pred_val_gb = gb_model.predict(X_val)

rmse_train_gb = np.sqrt(mean_squared_error(y_train, y_pred_train_gb))
rmse_val_gb = np.sqrt(mean_squared_error(y_val, y_pred_val_gb))

print("Gradient Boosting - TRAIN RMSE:", rmse_train_gb)
print("Gradient Boosting - VALIDATION RMSE:", rmse_val_gb)

'''
Linear Regression → TRAIN: -      | VALIDATION: 0.1271  
Random Forest     → TRAIN: 0.0526 | VALIDATION: 0.1443
Gradient Boosting → TRAIN: 0.0760 | VALIDATION: 0.1369
'''

gb_model_v2 = GradientBoostingRegressor(
    n_estimators=500,
    max_depth=3,
    learning_rate=0.02,
    random_state=42
)
gb_model_v2.fit(X_train, y_train)

y_pred_train_gb2 = gb_model_v2.predict(X_train)
y_pred_val_gb2 = gb_model_v2.predict(X_val)

rmse_train_gb2 = np.sqrt(mean_squared_error(y_train, y_pred_train_gb2))
rmse_val_gb2 = np.sqrt(mean_squared_error(y_val, y_pred_val_gb2))

print("GB v2 - TRAIN RMSE:", rmse_train_gb2)
print("GB v2 - VALIDATION RMSE:", rmse_val_gb2)

'''
Hiperparametre ayarıyla ağaç modellerini biraz iyileştirebiliyoruz
ama bu şekilde marjinal kazançlar elde ediyoruz. 
Sürekli parametre değiştirmek yerine, 
muhtemelen daha büyük etki yaratacak şey feature engineering
ve outlier temizliği — henüz hiç dokunmadığımız iki adım.
Bunlar genelde model seçiminden çok daha fazla fark yaratır.
'''

# En iyi modeli TÜM train verisiyle yeniden eğit (artık validation ayırmıyoruz)
final_model = GradientBoostingRegressor(
    n_estimators=500,
    max_depth=3,
    learning_rate=0.02,
    random_state=42
)
final_model.fit(X, y)

# Test verisi üzerinde tahmin yap
test_ids = test_encoded['Id']
X_test_final = test_encoded.drop(columns=['Id'])

test_predictions_log = final_model.predict(X_test_final)

# Log ölçeğinden gerçek dolar ölçeğine geri çevir
test_predictions = np.expm1(test_predictions_log)

# Kaggle formatında submission dosyası oluştur
submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': test_predictions
})

print(submission.head())
print(submission.shape)

SUBMISSIONS_DIR = BASE_DIR / 'submissions'
SUBMISSIONS_DIR.mkdir(exist_ok=True)  # klasör yoksa oluşturur, varsa hata vermez

submission.to_csv(SUBMISSIONS_DIR / 'submission_v1_gb.csv', index=False)
print("Kaydedildi:", SUBMISSIONS_DIR / 'submission_v1_gb.csv')


