import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('vgchartz-2024.csv')

print("Размерность датасета:", df.shape)

print("\nИнформация о датасете:")
print(df.info())

print("\nПервичные статистики:")
print(df.describe())

print("\nКоличество уникальных значений:")
print(df.nunique())

col = 'total_sales'
valid_col = df[col].dropna()
mean_val = np.mean(valid_col)
median_val = np.median(valid_col)
mode_res = stats.mode(valid_col, keepdims=True)
mode_val = mode_res.mode[0] if len(mode_res.mode) > 0 else np.nan

print(f"\nСтатистика по колонке '{col}':")
print(f"  Среднее значение: {mean_val:.4f}")
print(f"  Медиана: {median_val:.4f}")
print(f"  Мода: {mode_val}")

cat_columns = ['console', 'genre', 'publisher']
for cat in cat_columns:
    print(f"\nУникальные значения в столбце '{cat}':")
    print(df[cat].dropna().unique()[:10])

print("\nПропущенные значения")
missing_pct = (df.isnull().sum() / len(df)) * 100
print(missing_pct[missing_pct > 0])

for cat in cat_columns:
    df[cat] = df[cat].fillna('no data')

duplicates_count = df.duplicated().sum()
print(f"\nКоличество дубликатов до удаления: {duplicates_count}")
shape_before = df.shape
df.drop_duplicates(inplace=True)
print(f"Размерность после удаления дубликатов: {df.shape} (удалено: {shape_before[0] - df.shape[0]})")

valid_sales = df['total_sales'].dropna()
skew_val = stats.skew(valid_sales)
kurt_val = stats.kurtosis(valid_sales)
std_val = np.std(valid_sales, ddof=1)
mean_sales = np.mean(valid_sales)
median_sales = np.median(valid_sales)

summary_df = pd.DataFrame({
    'Метрика': ['Mean', 'Median', 'Std', 'Skewness', 'Kurtosis'],
    'Значение': [mean_sales, median_sales, std_val, skew_val, kurt_val]
})
print("\nСводная таблица статистик и формы распределения:")
print(summary_df.to_string(index=False))

cv_val = (std_val / mean_sales) * 100
print(f"\nКоэффициент вариации: {cv_val:.2f}%")

filtered_ps4 = df.query("console == 'PS4'").sort_values(by='total_sales', ascending=False)
print("\nТоп-3 игры на PS4:")
print(filtered_ps4[['title', 'console', 'total_sales']].head(3))

print("\nСрез Action:")
print(df.loc[df['genre'] == 'Action', ['title', 'genre', 'total_sales']].head(2))

print("\nСрез (строки 0-2, столбцы 1-3):")
print(df.iloc[0:3, 1:4])

genre_sales = df.groupby('genre')['total_sales'].sum().reset_index().sort_values(by='total_sales', ascending=False)
print("\nСуммарные продажи по жанрам:")
print(genre_sales.head(5))

top5_df = df.sort_values(by='total_sales', ascending=False)[['title', 'console', 'genre', 'publisher', 'total_sales']].head(5).reset_index(drop=True)
print("\nТоп-5 игр по глобальным продажам:")
print(top5_df.to_string())