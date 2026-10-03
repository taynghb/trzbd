import numpy as np

data_grades = np.array([
    68, 72, 55, 80, 91, 64, 77, 83, 59, 70,
    88, 62, 75, 95, 60, 100, 100, 100, 100, 100,
    67, 73, 58, 81, 90, 65, 78, 84, 61, 71,
    87, 63, 76, 94, 66, 100, 100, 100, 100, 100,
    69, 74, 56, 82, 92, 64, 79, 85, 60, 72,
    0, 0, 0, 0, 0, 65, 77, 83, 58, 70
])

A = np.array([68, 72, 55, 80, 91, 64, 77, 83, 59, 70, 88, 62, 75, 95, 60])

print("Среднее значение:", np.mean(data_grades))
print("Медиана:", np.median(data_grades))
print("Стандартное отклонение:", np.std(data_grades, ddof=0))
print("Дисперсия:", np.var(data_grades))
print("Максимум:", np.max(data_grades))
print("Минимум:", np.min(data_grades))
print("Размах:", np.max(data_grades) - np.min(data_grades))

print("\nПерцентили:")
print("25-й:", np.percentile(data_grades, 25))
print("50-й:", np.percentile(data_grades, 50))
print("75-й:", np.percentile(data_grades, 75))
print("90-й:", np.percentile(data_grades, 90))
print("99-й:", np.percentile(data_grades, 99))

q1 = np.quantile(data_grades, 0.25)
q3 = np.quantile(data_grades, 0.75)
iqr = q3 - q1
print("\nПервый квартиль (Q1):", q1)
print("Третий квартиль (Q3):", q3)
print("Межквартильный размах (IQR):", iqr)

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
outliers = data_grades[(data_grades < lower) | (data_grades > upper)]
print(f"\nНижняя граница: {lower}")
print(f"Верхняя граница: {upper}")
print("Выбросы:", outliers)

count_60_80 = np.sum((data_grades >= 60) & (data_grades <= 80))
print(f"\nСтудентов с отметками от 60 до 80 (включительно): {count_60_80}")

matrix = A.reshape(3, 5)
print("Матрица 3x5:\n", matrix)
print("Размерность (shape):", matrix.shape)
print("Общее количество элементов:", matrix.size)
print("Сумма по строкам (axis=1):", np.sum(matrix, axis=1))
print("Сумма по столбцам (axis=0):", np.sum(matrix, axis=0))
print("Транспонированная матрица:\n", matrix.T)

print("\nСтатистики по всем элементам матрицы:")
print("Среднее значение:", np.mean(matrix))
print("Стандартное отклонение:", np.std(matrix))
print("Дисперсия:", np.var(matrix))
print("Q1:", np.quantile(matrix, 0.25))
print("Q2 (медиана):", np.quantile(matrix, 0.50))
print("Q3:", np.quantile(matrix, 0.75))