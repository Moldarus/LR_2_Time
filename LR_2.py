import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных
df = pd.read_csv('passengers2.csv', parse_dates=['Month'])

# 1. Восполнение предыдущими значениями (forward fill)
df['target_ffill'] = df['target'].fillna(method='ffill')

# 2. Восполнение скользящим средним (window = 3)
# Вариант 1: вручную
df['target_rolling_manual'] = df['target'].copy()
for i in range(2, len(df)):
    if pd.isna(df.loc[i, 'target_rolling_manual']):
        df.loc[i, 'target_rolling_manual'] = df.loc[i-3:i-1, 'target'].mean()

# Вариант 2: с использованием rolling и mean
df['target_rolling_pandas'] = df['target'].rolling(window=3, min_periods=1).mean()

# 3. Интерполяция (линейная)
# Вариант 1: вручную с формулой интерполяции
df['target_interpol_manual'] = df['target'].copy()

for i in range(1, len(df)-1):
    if pd.isna(df.loc[i, 'target_interpol_manual']):
        x0, y0 = i-1, df.loc[i-1, 'target']
        x1, y1 = i+1, df.loc[i+1, 'target']
        x = i
        df.loc[i, 'target_interpol_manual'] = (y0 * (x1 - x) + y1 * (x - x0)) / (x1 - x0)

# Вариант 2: метод interpolate в pandas
df['target_interpol_pandas'] = df['target'].interpolate(method='linear')

# 4. Построение графиков для каждого способа
plt.figure(figsize=(14, 8))

# График для метода fillna (forward fill)
plt.subplot(2, 2, 1)
plt.plot(df['Month'], df['reference'], label='Reference', color='green')
plt.plot(df['Month'], df['target_ffill'], label='Target (forward fill)', color='blue')
plt.legend()
plt.title('Forward Fill Method')

# График для скользящего среднего (rolling mean)
plt.subplot(2, 2, 2)
plt.plot(df['Month'], df['reference'], label='Reference', color='green')
plt.plot(df['Month'], df['target_rolling_pandas'], label='Target (rolling mean)', color='orange')
plt.legend()
plt.title('Rolling Mean Method')

# График для линейной интерполяции (linear interpolation)
plt.subplot(2, 2, 3)
plt.plot(df['Month'], df['reference'], label='Reference', color='green')
plt.plot(df['Month'], df['target_interpol_pandas'], label='Target (interpolation)', color='red')
plt.legend()
plt.title('Linear Interpolation Method')

plt.tight_layout()
plt.show()

# 5. Дополнительное задание: Экспоненциальное сглаживание
d = 0.3  # Коэффициент сглаживания
df['smoothed'] = df['reference'].ewm(alpha=d).mean()

# Построение графика для сглаженных данных
plt.figure(figsize=(10, 6))
plt.plot(df['Month'], df['reference'], label='Reference', color='green')
plt.plot(df['Month'], df['smoothed'], label='Smoothed (exponential)', color='purple')
plt.legend()
plt.title('Exponential Smoothing of Reference Data')
plt.show()

# Сохранение результатов в отдельные файлы
df[['Month', 'target_ffill']].to_csv('target_ffill.csv', index=False)
df[['Month', 'target_rolling_pandas']].to_csv('target_rolling_pandas.csv', index=False)
df[['Month', 'target_interpol_pandas']].to_csv('target_interpol_pandas.csv', index=False)
df[['Month', 'smoothed']].to_csv('smoothed_reference.csv', index=False)
