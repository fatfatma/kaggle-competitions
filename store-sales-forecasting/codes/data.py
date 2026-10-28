import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


train = pd.read_csv('data/train.csv', parse_dates=['date'])
"""
print(train.shape)
print(train.head())
print(train.info())
"""

print("Missing values in each column:")
print(train.isnull().sum())

print("Date range in the dataset:")
print(train['date'].min(), "to", train['date'].max())

print("Number of unique stores:", train['store_nbr'].nunique())

print("Number of family types:", train['family'].nunique())
print(train['family'].unique())

#-------------------------------------
#sales istatistics

print(train['sales'].describe())
print()

print("sales distribution:", (train['sales'] == 0).mean())

#daily total sales GRAPH FOR visualization

daily_sales = train.groupby('date')['sales'].sum()
plt.figure(figsize=(12, 6))
plt.plot(daily_sales)
plt.title('Daily Total Sales')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.show()
#________________________
#EVERY FAMILY SALES STATISTICS 

family_stats = train.groupby('family')['sales'].agg(['sum', 'mean', 'median']).sort_values('sum', ascending=False)
print(family_stats)

#----------------------------------
# Her (store_nbr, family) kombinasyonunun toplam satışını hesapla
store_family_sum = train.groupby(['store_nbr', 'family'])['sales'].sum().reset_index()

# Toplam satışı 0 olan kombinasyon var mı?
zero_combos = store_family_sum[store_family_sum['sales'] == 0]
print("Hiç satış olmayan (mağaza, ürün ailesi) kombinasyon sayısı:", len(zero_combos))
print(zero_combos)


