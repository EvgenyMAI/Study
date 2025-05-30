import numpy as np
import matplotlib.pyplot as plt

# Правая часть уравнения
def rhs(x, y, dy):
    return 2 * y * (1 + np.tan(x) ** 2)

# Точное решение
def analytical(x):
    return -np.tan(x)

# Метод Рунге–Кутты 4-го порядка
# Решает задачу Коши методом Рунге–Кутты (используется внутри метода стрельбы).
def rk4_solver(f, x_start, y_start, dy_start, step, steps):
    x_data, y_data, dy_data = [x_start], [y_start], [dy_start]
    for _ in range(steps):
        x, y, dy = x_data[-1], y_data[-1], dy_data[-1]

        k1 = step * dy
        l1 = step * f(x, y, dy)

        k2 = step * (dy + l1 / 2)
        l2 = step * f(x + step / 2, y + k1 / 2, dy + l1 / 2)

        k3 = step * (dy + l2 / 2)
        l3 = step * f(x + step / 2, y + k2 / 2, dy + l2 / 2)

        k4 = step * (dy + l3)
        l4 = step * f(x + step, y + k3, dy + l3)

        y_next = y + (k1 + 2*k2 + 2*k3 + k4) / 6
        dy_next = dy + (l1 + 2*l2 + 2*l3 + l4) / 6

        x_data.append(x + step)
        y_data.append(y_next)
        dy_data.append(dy_next)

    return x_data, y_data, dy_data

# Метод стрельбы
# Подбирает нужное начальное значение производной, чтобы в конце отрезка получить 𝑦(𝜋/6)=−3^(1/2)/3
def shooting(f, x0, x1, y_start, y_end, h, guess1, guess2, tol=1e-8):
    n_steps = int((x1 - x0) / h)

    def boundary_miss(eta):
        _, y, _ = rk4_solver(f, x0, y_start, eta, h, n_steps)
        return y[-1] - y_end

    eta_prev, eta_curr = guess1, guess2
    while True:
        # Для каждого значения решаем задачу Коши методом Рунге–Кутты
        val_prev, val_curr = boundary_miss(eta_prev), boundary_miss(eta_curr)
        # Смотрим, насколько далеко мы промахнулись в точке x=b
        if abs(val_curr) < tol or val_curr == val_prev:
            break
        # Используем метод секущих для подбора нового приближения
        eta_next = eta_curr - val_curr * (eta_curr - eta_prev) / (val_curr - val_prev)
        eta_prev, eta_curr = eta_curr, eta_next

    x_vals, y_vals, _ = rk4_solver(f, x0, y_start, eta_curr, h, n_steps)
    return x_vals, y_vals

# Конечно-разностный метод
# Заменяет производные на разности — создает систему линейных уравнений и решает её
def difference_scheme(x0, x1, y0, yN, n):
    h = (x1 - x0) / n
    # Разбиваем отрезок [a,b] на n узлов:
    x_points = np.linspace(x0, x1, n + 1)

    # Формируем трёхдиагональную матрицу коэффициентов
    main_diag = -2 / h**2 + 2 * (1 + np.tan(x_points[1:-1])**2)
    lower_diag = upper_diag = np.ones(n - 1) / h**2
    rhs_vector = np.zeros(n - 1)
    rhs_vector[0] -= y0 / h**2
    rhs_vector[-1] -= yN / h**2

    alpha = np.zeros(n - 1)
    beta = np.zeros(n - 1)

    # Решаем систему линейных уравнений методом прогонки
    alpha[0] = -upper_diag[0] / main_diag[0]
    beta[0] = rhs_vector[0] / main_diag[0]
    for i in range(1, n - 1):
        denom = main_diag[i] + lower_diag[i] * alpha[i - 1]
        alpha[i] = -upper_diag[i] / denom
        beta[i] = (rhs_vector[i] - lower_diag[i] * beta[i - 1]) / denom

    y_sol = np.zeros(n + 1)
    y_sol[0], y_sol[-1] = y0, yN
    y_sol[-2] = beta[-1]
    for i in range(n - 3, -1, -1):
        y_sol[i + 1] = alpha[i] * y_sol[i + 2] + beta[i]

    return x_points, y_sol

# Метод Рунге–Ромберга
# Оценивает погрешность: запускает метод с двумя шагами h и h/2, и сравнивает результат.
def rr_error(y_coarse, y_fine, order):
    y_c = np.array(y_coarse)
    y_f = np.array(y_fine)[::2][:len(y_c)]
    err_final = abs((y_f[-1] - y_c[-1]) / (2**order - 1))
    err_max = np.max(np.abs((y_f - y_c) / (2**order - 1)))
    return err_final, err_max

# Функция визуализации
def plot_solution(x_num, y_num, method_name, x_exact=None, y_exact=None):
    plt.figure(figsize=(10, 5))
    if x_exact is not None and y_exact is not None:
        plt.plot(x_exact, y_exact, 'k-', label='Точное решение', linewidth=2)
    plt.plot(x_num, y_num, 'o--', label=method_name, markersize=4)
    plt.title(f'Решение методом: {method_name}')
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    # Границы и условия
    a, b = 0, np.pi / 6
    ya, yb = 0, -np.sqrt(3) / 3
    h = 0.01
    n = int((b - a) / h)

    # Решения
    x_shot, y_shot = shooting(rhs, a, b, ya, yb, h, -1.0, -2.0)
    x_fd, y_fd = difference_scheme(a, b, ya, yb, n)

    # Более точные решения для Рунге–Ромберга
    x_shot_fine, y_shot_fine = shooting(rhs, a, b, ya, yb, h / 2, -1.0, -2.0)
    x_fd_fine, y_fd_fine = difference_scheme(a, b, ya, yb, 2 * n)

    # Оценка погрешностей
    err_shot_end, err_shot_max = rr_error(y_shot, y_shot_fine, 4)
    err_fd_end, err_fd_max = rr_error(y_fd, y_fd_fine, 2)

    print("== Метод стрельбы ==")
    print(f"Погрешность в последней точке: {err_shot_end:.3e}")
    print(f"Максимальная погрешность:     {err_shot_max:.3e}\n")

    print("== Конечно-разностный метод ==")
    print(f"Погрешность в последней точке: {err_fd_end:.3e}")
    print(f"Максимальная погрешность:      {err_fd_max:.3e}")

    # Построение графиков
    x_true = np.linspace(a, b, 500)
    y_true = analytical(x_true)

    plot_solution(x_shot, y_shot, method_name="Метод стрельбы", x_exact=x_true, y_exact=y_true)
    plot_solution(x_fd, y_fd, method_name="Конечно-разностный метод", x_exact=x_true, y_exact=y_true)

if __name__ == "__main__":
    main()