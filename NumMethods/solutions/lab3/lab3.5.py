import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.interpolate import BarycentricInterpolator

def f(x):
    """
    Возвращает заданную функцию
    """
    return x / (x**4 + 81)

def exact_integral():
    """
    Вычисляет точное значение определённого интеграла с помощью библиотеки SymPy
    """
    x = sp.symbols('x')
    integrand = x / (x**4 + 81)
    result = sp.integrate(integrand, (x, 0, 2))
    return float(result.evalf())

def generate_x(a, b, h):
    """
    Генерирует массив точек на отрезке [a, b] с шагом h
    """
    n = int((b - a) / h)
    return np.linspace(a, b, n + 1)

def evaluate_function(x):
    """
    Вычисляет значения функции f(x) в каждой точке массива x
    """
    return f(x)

def rectangle_rule(x, h):
    """
    Вычисляет определённый интеграл методом прямоугольников.
    Использует средние точки на каждом интервале
    """
    sum_val = 0.0
    for i in range(len(x) - 1):
        mid = (x[i] + x[i + 1]) / 2
        sum_val += f(mid)
    return h * sum_val

def trapezoid_rule(y, h):
    """
    Вычисляет определённый интеграл методом трапеций
    """
    return h * (y[0] / 2 + y[-1] / 2 + np.sum(y[1:-1]))

def simpson_rule(y, h):
    """
    Вычисляет определённый интеграл методом Симпсона
    (работает только при чётном числе интервалов)
    """
    n = len(y)
    if (n - 1) % 2 != 0:
        print("Ошибка: нечетное число интервалов для Симпсона.")
        return float('nan')
    sum_val = y[0] + y[-1]
    for i in range(1, n - 1):
        sum_val += 4 * y[i] if i % 2 != 0 else 2 * y[i]
    return h / 3 * sum_val

def rect_residual(a, b, h):
    """
    Оценивает остаточный член для метода прямоугольников
    """
    M2 = 0.01  # максимум |f''(x)| на [a, b], оценка вручную
    return (M2 * (b - a) * h**2) / 24

def trap_residual(a, b, h):
    """
    Оценивает остаточный член для метода трапеций
    """
    M2 = 0.01
    return (M2 * (b - a) * h**2) / 12

def simpson_residual(a, b, h):
    """
    Оценивает остаточный член для метода Симпсона
    """
    M4 = 0.05  # оценка производной f^(4)(x)
    return (M4 * (b - a)**5) / (180 * ((b - a)/h)**4)

def runge_romberg(method, a, b, h1, h2, p):
    """
    Вычисляет уточнённое значение интеграла и абсолютную погрешность по методу Рунге–Ромберга
    """
    x1 = generate_x(a, b, h1)
    x2 = generate_x(a, b, h2)
    y1 = evaluate_function(x1)
    y2 = evaluate_function(x2)

    if method == "Прямоугольники":
        I1 = rectangle_rule(x1, h1)
        I2 = rectangle_rule(x2, h2)
    elif method == "Трапеции":
        I1 = trapezoid_rule(y1, h1)
        I2 = trapezoid_rule(y2, h2)
    elif method == "Симпсон":
        I1 = simpson_rule(y1, h1)
        I2 = simpson_rule(y2, h2)
    else:
        raise ValueError("Неизвестный метод")

    k = h2 / h1
    refined = I1 + (I1 - I2) / (k**p - 1)
    error = abs(refined - exact_integral())

    print(f"{method} → уточнённое значение: {refined:.8f}, абсолютная погрешность: {error:.8f}")

# ========== ГРАФИКИ ==========
def plot_function(a, b):
    """
    Строит график функции f(x)
    """
    x_vals = np.linspace(a, b, 500)
    y_vals = f(x_vals)
    plt.figure(figsize=(8, 4))
    plt.plot(x_vals, y_vals, label='f(x)', color='blue')
    plt.title('График функции f(x)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_rectangle_method(x, h):
    """
    Визуализирует метод прямоугольников
    """
    plt.figure(figsize=(8, 4))
    x_vals = np.linspace(x[0], x[-1], 500)
    plt.plot(x_vals, f(x_vals), 'b', label='f(x)')
    for i in range(len(x) - 1):
        mid = (x[i] + x[i + 1]) / 2
        height = f(mid)
        plt.bar(mid, height, width=h, align='center', alpha=0.4, edgecolor='black')
    plt.title("Метод прямоугольников")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_trapezoid_method(x, y):
    """
    Визуализирует метод трапеций
    """
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, 'b', label='f(x)')
    for i in range(len(x) - 1):
        plt.fill([x[i], x[i], x[i+1], x[i+1]],
                 [0, y[i], y[i+1], 0],
                 'orange', edgecolor='black', alpha=0.4)
    plt.title("Метод трапеций")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_simpson_method(x, y):
    """
    Строит график парабол, аппроксимирующих f(x) по методу Симпсона
    """
    plt.figure(figsize=(10, 5))
    plt.title("Метод Симпсона: приближение параболами")
    plt.xlabel("x")
    plt.ylabel("f(x)")

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, (len(x) - 1) // 2))

    # Для каждой тройки узлов строим параболу и закрашиваем площадь под ней
    for i in range(0, len(x) - 2, 2):
        xi = x[i:i+3]
        yi = y[i:i+3]

        # Интерполяция параболой через 3 точки
        parabola = BarycentricInterpolator(xi, yi)

        xf = np.linspace(xi[0], xi[-1], 100)
        yf = parabola(xf)

        plt.plot(xf, yf, color=colors[i // 2], label=f"Парабола [{xi[0]:.2f}, {xi[-1]:.2f}]")
        plt.fill_between(xf, yf, alpha=0.2, color=colors[i // 2])

        # Узлы параболы
        plt.plot(xi, yi, 'ko')

    plt.legend(loc="upper right", fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ========== MAIN ==========
if __name__ == "__main__":
    a, b = 0.0, 2.0
    h1 = 0.5
    h2 = 0.25
    steps = [h1, h2]

    print("==== Визуализация функции ====")
    plot_function(a, b)

    for h in steps:
        print(f"\n==== Шаг h = {h} ====")
        x = generate_x(a, b, h)
        y = evaluate_function(x)

        rect = rectangle_rule(x, h)
        trap = trapezoid_rule(y, h)
        simp = simpson_rule(y, h)

        print(f"Прямоугольники: {rect:.8f}")
        print(f"Трапеции:       {trap:.8f}")
        print(f"Симпсон:        {simp:.8f}")

        print(f"Оценка остатка (Rect):     {rect_residual(a, b, h):.8f}")
        print(f"Оценка остатка (Trap):     {trap_residual(a, b, h):.8f}")
        if (len(y) - 1) % 2 == 0:
            print(f"Оценка остатка (Simpson):  {simpson_residual(a, b, h):.8f}")

        # Построение графиков
        plot_rectangle_method(x, h)
        plot_trapezoid_method(x, y)
        if (len(y) - 1) % 2 == 0:
            plot_simpson_method(x, y)

    print("\n==== Метод Рунге–Ромберга–Ричардсона ====")
    runge_romberg("Прямоугольники", a, b, h2, h1, 2)
    runge_romberg("Трапеции", a, b, h2, h1, 2)
    runge_romberg("Симпсон", a, b, h2, h1, 4)

    print("\n==== Точное значение интеграла ====")
    print(f"Точное значение: {exact_integral():.8f}")