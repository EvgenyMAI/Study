import numpy as np
import matplotlib.pyplot as plt

# Правая часть системы (y'' = f(x, y, y')) для уравнения: xy'' + y' = 0 => y'' = -y'/x

def f(x, y, z):
    return -z / x

# Точное решение
exact_solution = lambda x: 1 + np.log(np.abs(x))

# Метод Эйлера
def euler_method(f, x0, y0, z0, h, n):
    x_vals, y_vals, z_vals = [x0], [y0], [z0]
    for _ in range(n):
        x, y, z = x_vals[-1], y_vals[-1], z_vals[-1]
        x_next = x + h
        y_next = y + h * z
        z_next = z + h * f(x, y, z)
        x_vals.append(x_next)
        y_vals.append(y_next)
        z_vals.append(z_next)
    return x_vals, y_vals

# Метод Рунге-Кутты 4-го порядка
def runge_kutta_4(f, x0, y0, z0, h, n):
    x_vals, y_vals, z_vals = [x0], [y0], [z0]
    for _ in range(n):
        x, y, z = x_vals[-1], y_vals[-1], z_vals[-1]
        
        k1 = h * z
        l1 = h * f(x, y, z)

        k2 = h * (z + l1 / 2)
        l2 = h * f(x + h / 2, y + k1 / 2, z + l1 / 2)

        k3 = h * (z + l2 / 2)
        l3 = h * f(x + h / 2, y + k2 / 2, z + l2 / 2)

        k4 = h * (z + l3)
        l4 = h * f(x + h, y + k3, z + l3)

        x_vals.append(x + h)
        y_vals.append(y + (k1 + 2*k2 + 2*k3 + k4) / 6)
        z_vals.append(z + (l1 + 2*l2 + 2*l3 + l4) / 6)
    return x_vals, y_vals, z_vals

# Метод Адамса 4-го порядка
def adams_method(f, x0, y0, z0, h, n):
    x_vals, y_vals, z_vals = runge_kutta_4(f, x0, y0, z0, h, 3)

    for i in range(3, n):
        x0, x1, x2, x3 = x_vals[-4:]
        y0, y1, y2, y3 = y_vals[-4:]
        z0, z1, z2, z3 = z_vals[-4:]

        f0 = z0
        f1 = z1
        f2 = z2
        f3 = z3

        g0 = f(x0, y0, z0)
        g1 = f(x1, y1, z1)
        g2 = f(x2, y2, z2)
        g3 = f(x3, y3, z3)

        y_next = y3 + h * (55*f3 - 59*f2 + 37*f1 - 9*f0) / 24
        z_next = z3 + h * (55*g3 - 59*g2 + 37*g1 - 9*g0) / 24

        x_vals.append(x3 + h)
        y_vals.append(y_next)
        z_vals.append(z_next)

    return x_vals, y_vals

# Метод Рунге–Ромберга для оценки погрешности
def runge_romberg(y_h, y_h2, p):
    y_h2_thinned = y_h2[::2]
    return [(y2 - y1) / (2**p - 1) for y1, y2 in zip(y_h, y_h2_thinned)]

# Отрисовка отдельных графиков для каждого метода
def plots(x_exact, y_exact, results):
    for label, (x_vals, y_vals) in results.items():
        plt.figure(figsize=(8, 5))
        plt.plot(x_exact, y_exact, label='Точное решение', linewidth=2)
        plt.plot(x_vals, y_vals, '--o', label=label, alpha=0.75)
        plt.title(f"Сравнение: {label} и точное решение")
        plt.xlabel("x")
        plt.ylabel("y(x)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

def main():
    x0, y0, z0 = 1, 1, 1
    h = 0.1
    n = int((2 - x0) / h)

    x_euler, y_euler = euler_method(f, x0, y0, z0, h, n)
    x_rk, y_rk, _ = runge_kutta_4(f, x0, y0, z0, h, n)
    x_adams, y_adams = adams_method(f, x0, y0, z0, h, n)

    h2 = h / 2
    n2 = 2 * n
    _, y_euler_h2 = euler_method(f, x0, y0, z0, h2, n2)
    _, y_rk_h2, _ = runge_kutta_4(f, x0, y0, z0, h2, n2)
    _, y_adams_h2 = adams_method(f, x0, y0, z0, h2, n2)

    err_euler = runge_romberg(y_euler, y_euler_h2, p=1)
    err_rk = runge_romberg(y_rk, y_rk_h2, p=4)
    err_adams = runge_romberg(y_adams, y_adams_h2, p=4)

    print("\nОценка ошибки (метод Рунге–Ромберга) в x = 2.0:")
    print(f"Эйлер (порядок 1):           {err_euler[-1]:.8f}")
    print(f"Рунге–Кутта 4 порядка:       {err_rk[-1]:.8f}")
    print(f"Адамс–Бэшфорт 4 порядка:     {err_adams[-1]:.8f}")

    x_true = np.linspace(1, 2, 100)
    y_true = exact_solution(x_true)

    plots(x_true, y_true, {
        'Метод Эйлера': (x_euler, y_euler),
        'Метод Рунге-Кутты 4': (x_rk, y_rk),
        'Метод Адамса 4': (x_adams, y_adams)
    })

if __name__ == "__main__":
    main()