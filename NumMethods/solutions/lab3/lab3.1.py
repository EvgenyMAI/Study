import numpy as np

# Функция f(x) = cot(x) + x
def f(x):
    return 1 / np.tan(x) + x

# Интерполяционный многочлен Лагранжа
def lagrange(xi, yi, x):
    n = len(xi)

    # omega(x) = произведение (x - xi)
    omega = 1.0
    for i in range(n):
        omega *= (x - xi[i])

    result = 0.0
    for i in range(n):
        omega_prime = 1.0
        for j in range(n):
            if j != i:
                omega_prime *= (xi[i] - xi[j])
        result += (yi[i] / omega_prime) * (omega / (x - xi[i]))

    return result

# Таблица разделённых разностей (для метода Ньютона)
def divided_differences(xi, yi):
    n = len(xi)
    table = np.zeros((n, n))
    table[:, 0] = yi

    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i][j - 1] - table[i + 1][j - 1]) / (xi[i] - xi[i + j])

    return table

# Интерполяционный многочлен Ньютона
def newton(xi, dd_table, x):
    result = dd_table[0][0]
    product = 1.0
    for i in range(1, len(xi)):
        product *= (x - xi[i - 1])
        result += dd_table[0][i] * product
    return result

# Абсолютная погрешность
def absolute_error(real_value, approx_value):
    return abs(real_value - approx_value)

# Основной метод решения задачи
def solve(xi, x_star, label):
    print(f"\nРешение для набора {label}:")
    yi = [f(x) for x in xi]

    print("Точки (X, Y):")
    for i, (x_val, y_val) in enumerate(zip(xi, yi)):
        print(f"x[{i}] = {x_val:.4f}, y[{i}] = {y_val:.6f}")

    lagrange_value = lagrange(xi, yi, x_star)
    dd_table = divided_differences(xi, yi)
    newton_value = newton(xi, dd_table, x_star)
    real_value = f(x_star)

    print(f"\nВычисления в точке X* = {x_star}:")
    print(f"Истинное значение: f({x_star:.2f}) = {real_value:.6f}")
    print(f"Лагранжев интерполянт: L({x_star:.2f}) = {lagrange_value:.6f}")
    print(f"Ньютонов интерполянт: N({x_star:.2f}) = {newton_value:.6f}")

    print("\nАбсолютные погрешности:")
    print(f"|f(x*) - L(x*)| = {absolute_error(real_value, lagrange_value):.6f}")
    print(f"|f(x*) - N(x*)| = {absolute_error(real_value, newton_value):.6f}")

if __name__ == "__main__":
    pi = np.pi
    xi_a = [pi/8, 2*pi/8, 3*pi/8, 4*pi/8]
    xi_b = [pi/8, pi/3, 3*pi/8, pi/2]
    x_star = 3*pi/16

    solve(xi_a, x_star, "а) {π/8, 2π/8, 3π/8, 4π/8}")
    solve(xi_b, x_star, "б) {π/8, π/3, 3π/8, π/2}")