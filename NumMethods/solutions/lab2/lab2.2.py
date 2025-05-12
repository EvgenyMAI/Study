import math
import numpy as np

EPSILON = 1e-6
MAX_ITER = 100

# Уравнения системы
def f1(x1, x2):
    return 2 * x1 ** 2 - x2 + x2 ** 2 - 2

def f2(x1, x2):
    return x1 - math.sqrt(x2 + 2) + 1

# Фи-функции для метода простой итерации
def phi1(x1, x2):
    return math.sqrt(x2 + 2) - 1

def phi2(x1, x2):
    return 2 * x1 ** 2 + x2 ** 2 - 2

# Производные фи-функций
def dphi1_dx2(x2):
    return 1 / (2 * math.sqrt(x2 + 2))

def dphi2_dx1(x1):
    return 4 * x1

def dphi2_dx2(x2):
    return 2 * x2

# Поиск приближённого пересечения - Ищем такую точку, чтобы значения обеих функций в этой точке были близки к 0
def find_root_interval(from_, to, step):
    minD = 0.1
    best = None

    x_range = np.arange(from_, to + step, step)
    for x1 in x_range:
        for x2 in x_range:
            d = math.sqrt(f1(x1, x2)**2 + f2(x1, x2)**2)
            if d < minD:
                minD = d
                best = (x1, x2)

    if minD < 0.1:
        print(f"Найдено приближённое пересечение: x1 = {best[0]:.4f}, x2 = {best[1]:.4f}, D = {minD:.6f}")
        return best
    else:
        print(f"Пересечения не найдено в диапазоне [{from_}; {to}], minD = {minD:.6f}")
        return None

# Проверка условия сходимости
def check_convergence(x1, x2):
    J = [
        [0, dphi1_dx2(x2)],
        [dphi2_dx1(x1), dphi2_dx2(x2)]
    ]

    # Условие сходимости: вектор функция phi непрерывна вместе со своей производной в области G и max||phi'(x)|| <= q < 1.

    max_row_sum = max(sum(abs(j) for j in row) for row in J)
    print(f"Проверка сходимости: q = {max_row_sum:.4f} {'< 1 — сходимость есть' if max_row_sum < 1 else '>= 1 — сходимости НЕТ'}")
    return max_row_sum < 1

# Метод простой итерации
def simple_iteration(x1_0, x2_0):
    x1, x2 = x1_0, x2_0
    iter_count = 0

    if not check_convergence(x1, x2):
        print("Метод простой итерации может не сойтись.")

    while iter_count < MAX_ITER:
        x1_new = phi1(x1, x2)
        x2_new = phi2(x1, x2)

        error = max(abs(x1_new - x1), abs(x2_new - x2))
        if error < EPSILON:
            break

        x1, x2 = x1_new, x2_new
        iter_count += 1

    print(f"Простая итерация: x1 = {x1:.6f}, x2 = {x2:.6f} за {iter_count} итераций")

# Метод Ньютона
def newton(x1_0, x2_0):
    x1, x2 = x1_0, x2_0
    iter_count = 0

    while iter_count < MAX_ITER:
        f1v = f1(x1, x2)
        f2v = f2(x1, x2)

        df1_dx1 = 4 * x1
        df1_dx2 = -1 + 2 * x2
        df2_dx1 = 1
        df2_dx2 = -1 / (2 * math.sqrt(x2 + 2))

        detJ = df1_dx1 * df2_dx2 - df1_dx2 * df2_dx1
        detA1 = f1v * df2_dx2 - df1_dx2 * f2v
        detA2 = df1_dx1 * f2v - f1v * df2_dx1

        if abs(detJ) < 1e-12:
            print("Матрица Якоби вырождена, метод Ньютона остановлен.")
            return
        
        # xn(k+1) = xn(k) - detAn(k)/detJ(k), для каждого xn
        x1_new = x1 - detA1 / detJ
        x2_new = x2 - detA2 / detJ

        # Условие окончания: ||x(k-1)=x(k)|| = max по i|xi(k+1) - xi(k)|.
        error = max(abs(x1_new - x1), abs(x2_new - x2))
        if error < EPSILON:
            break

        x1, x2 = x1_new, x2_new
        iter_count += 1

    print(f"Ньютон: x1 = {x1:.6f}, x2 = {x2:.6f} за {iter_count} итераций")

# Главная функция
def main():
    interval = find_root_interval(0, 2, 0.1)
    if interval is None:
        print("Не удалось найти интервал, содержащий корень.")
        return

    x1_0, x2_0 = interval
    print(f"Найдено начальное приближение: x1 = {x1_0:.4f}, x2 = {x2_0:.4f}")

    simple_iteration(x1_0, x2_0)
    newton(x1_0, x2_0)

if __name__ == "__main__":
    main()