import numpy as np
import matplotlib.pyplot as plt

def solve_tridiagonal(a, b, c, d):
    """Решает трёхдиагональную систему линейных уравнений методом прогонки.
        Шаги:
        - Прямой ход — преобразуем систему к верхнетреугольному виду;
        - Обратный ход — находим решения xi по формулам.
    """
    n = len(b)
    P = np.zeros(n)
    Q = np.zeros(n)
    x = np.zeros(n)

    P[0] = -c[0] / b[0]
    Q[0] = d[0] / b[0]

    # Прямой ход
    for i in range(1, n):
        denom = b[i] + a[i] * P[i - 1]
        P[i] = 0 if i == n - 1 else -c[i] / denom
        Q[i] = (d[i] - a[i] * Q[i - 1]) / denom

   # Обратный ход
    x[-1] = Q[-1]
    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]

    return x

def cubic_spline_interpolation(x, f, x_star):
    """
    Строит кубический сплайн по точкам и вычисляет значение сплайна в заданной точке X*.
    """
    n = len(x) - 1
    h = np.zeros(n + 1)
    for i in range(1, n + 1):
        h[i] = x[i] - x[i - 1]

    # Формирование СЛАУ для коэффициентов c (вторая производная сплайна)
    A = np.zeros(n - 1)
    B = np.zeros(n - 1)
    C = np.zeros(n - 1)
    D = np.zeros(n - 1)

    for i in range(2, n + 1):
        idx = i - 2
        A[idx] = 0.0 if i == 2 else h[i - 1]
        B[idx] = 2 * (h[i - 1] + h[i])
        C[idx] = 0.0 if i == n else h[i]
        D[idx] = 3 * ((f[i] - f[i - 1]) / h[i] - (f[i - 1] - f[i - 2]) / h[i - 1])

    solutionC = solve_tridiagonal(A, B, C, D)

    c = np.zeros(n + 2)  # индексируем с 1 до n + 1
    c[1] = 0.0
    c[n + 1] = 0.0
    for i in range(2, n + 1):
        c[i] = solutionC[i - 2]

    a = np.zeros(n + 1)
    b = np.zeros(n + 1)
    d = np.zeros(n + 1)

    # Вычисление всех коэффициентов сплайна
    for i in range(1, n + 1):
        a[i] = f[i - 1]
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])
        b[i] = (f[i] - f[i - 1]) / h[i] - (h[i] / 3) * (2 * c[i] + c[i + 1])

    # Определение нужного интервала для x*
    interval = 1
    for i in range(1, n + 1):
        if x_star >= x[i - 1] and x_star <= x[i]:
            interval = i
            break
    
    # Вычисление значения сплайна в x*
    dx = x_star - x[interval - 1]
    result = a[interval] + b[interval] * dx + c[interval] * dx**2 + d[interval] * dx**3

    print(f"Значение функции в точке X* = {x_star:.3f}: {result:.6f}")
    for i in range(1, n + 1):
        print(f"[{x[i-1]:.1f}, {x[i]:.1f}]: a={a[i]:.6f}, b={b[i]:.6f}, c={c[i]:.6f}, d={d[i]:.6f}")

     # Построение графика сплайна
    spline_x = []
    spline_y = []
    for i in range(1, n + 1):
        xi = x[i - 1]
        xi_next = x[i]
        dx_range = np.linspace(0, xi_next - xi, 100)
        yi = a[i] + b[i] * dx_range + c[i] * dx_range**2 + d[i] * dx_range**3
        spline_x.extend(xi + dx_range)
        spline_y.extend(yi)

    plt.figure(figsize=(8, 5))
    plt.plot(spline_x, spline_y, label='Кубический сплайн', color='blue')
    plt.plot(x, f, 'o', label='Узлы интерполяции', color='red')
    plt.plot(x_star, result, 's', label=f'S(x*) = {result:.3f}', color='green')
    plt.title('Интерполяция кубическим сплайном')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.legend()
    plt.show()

    return result

if __name__ == "__main__":
    x_vals = [0.1, 0.5, 0.9, 1.3, 1.7]
    f_vals = [100.01, 4.25, 2.0446, 2.2817, 3.2360]
    x_star = 0.8

    cubic_spline_interpolation(x_vals, f_vals, x_star)