import numpy as np

a = np.array([0, -7, 4, -4, -1], dtype=float)  # Нижняя диагональ (a1 = 0)
b = np.array([16, -16, 12, 12, 7], dtype=float)  # Главная диагональ
c = np.array([-8, 5, 3, -7, 0], dtype=float)  # Верхняя диагональ (cn = 0)
d = np.array([0, -123, -68, 104, 20], dtype=float)  # Правая часть

n = len(b)

is_stable = True
for i in range(n):
    ai = a[i] if i > 0 else 0
    ci = c[i] if i < n - 1 else 0
    if abs(b[i]) < abs(ai) + abs(ci):
        is_stable = False
        print(f"Условие устойчивости нарушено на i={i+1}: |b[{i}]| < |a[{i}]| + |c[{i}]|")

if is_stable:
    print("Матрица удовлетворяет условиям устойчивости.")
    
    # Прямой ход
    P = np.zeros(n, dtype=float)
    Q = np.zeros(n, dtype=float)
    x = np.zeros(n, dtype=float)

    P[0] = -c[0] / b[0]
    Q[0] = d[0] / b[0]
    for i in range(1, n):
        denominator = b[i] + a[i] * P[i - 1]
        P[i] = -c[i] / denominator if i < n - 1 else 0  # Учитываем, что cn = 0
        Q[i] = (d[i] - a[i] * Q[i - 1]) / denominator

    # Обратный ход
    x[-1] = Q[-1]  # Т.к. cn = 0
    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]

    print("Решение СЛАУ:")
    for i, xi in enumerate(x, 1):
        print(f"x{i} = {xi:.6f}")
else:
    print("Решение не выполняется из-за нарушения условий устойчивости.")