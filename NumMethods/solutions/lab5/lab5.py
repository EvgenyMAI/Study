import numpy as np
import matplotlib.pyplot as plt
import math

a_coef = 1.0 # коэффициент теплопроводности (a > 0)
L = math.pi # длина стержня (x принадлежит [0, π])

def g_left(t):
    return math.exp(-a_coef * t) # поток на левой границе: u_x(0,t) = e^{-a t}

def g_right(t):
    return -math.exp(-a_coef * t) # поток на правой границе: u_x(π,t) = -e^{-a t}

def f(x, t):
    return 0.0 # правая часть отсутствует

def exact(x, t):
    return math.exp(-a_coef * t) * math.sin(x) # аналитическое решение

# Метод прогонки (Томаса)
def thomas_solve(a, b, c, d):
    # Метод решения трёхдиагональной СЛАУ (используется в неявных схемах)
    n = len(b)
    cp = np.zeros(n)
    dp = np.zeros(n)
    x = np.zeros(n)

    # Прямой ход прогонки
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i] * cp[i - 1]
        denom = denom if abs(denom) > 1e-16 else 1e-16  # защита от деления на 0
        cp[i] = 0.0 if i == n - 1 else c[i] / denom
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom

    # Обратный ход (нахождение решения)
    x[-1] = dp[-1]
    for i in reversed(range(n - 1)):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x

# Функции для построения графиков
def create_chart(x, numerical, exact_values, scheme_name, T):
    # Столбчатая диаграмма ошибок для сравнения схем и аппроксимаций
    plt.figure(figsize=(7, 5))
    plt.plot(x, numerical, 'r-', label=f'Численное решение ({scheme_name})')
    plt.plot(x, exact_values, 'k--', label='Точное решение')
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title(f'Сравнение решений: {scheme_name} (t={T})')
    plt.legend()
    plt.grid(True)
    plt.show()

def create_error_chart(max_errors, l2_errors, labels):
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, max_errors, width, label='Максимальная ошибка')
    plt.bar(x + width / 2, l2_errors, width, label='L2 ошибка')
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylabel('Ошибка')
    plt.title('Сравнение ошибок по методам')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Аппроксимация правого граничного условия
def apply_boundary_condition(u, N, h, t, bc_option, u_old, tau):
    """Возвращает значение u[N] на правой границе по выбранной аппроксимации"""
    gk1 = g_right(t) # значение потока справа в момент t

    # Первая аппроксимация (первая точность)
    if bc_option == 1:
        return u[N - 1] + h * gk1
    # Вторая аппроксимация (вторая точность)
    elif bc_option == 2:
        if N >= 2:
            return (4 * u[N - 1] - u[N - 2] + 2 * h * gk1) / 3
        else:
            return u[N - 1] + h * gk1
    # Третья аппроксимация (повышенной точности с учётом tau)
    else:
        fNk1 = f(L, t)
        den = 1 + (h * h) / (2 * tau)
        num = u[N - 1] + h * gk1 + (h * h / (2 * tau)) * u_old[N] + (h * h / 2) * fNk1
        return num / den

# Явная схема
def explicit_step(u_old, u_new, N, h, tau, t_k, t_k1, bc_option):
    x = np.linspace(0, L, N + 1)
    for j in range(1, N):
        # Вычисляем вторую производную по x (лапласиан)
        lap = (u_old[j + 1] - 2 * u_old[j] + u_old[j - 1]) / (h * h)
        rhs = f(x[j], t_k) # правая часть (здесь = 0)
        u_new[j] = u_old[j] + a_coef * tau * (lap + rhs)

    # Левая граница: u_x(0,t) = e^{-a t} => u_0 = u_1 - h * g_left
    u_new[0] = u_new[1] - h * g_left(t_k1)

    # Правая граница через функцию аппроксимации (один из BC1–BC3)
    u_new[N] = apply_boundary_condition(u_new, N, h, t_k1, bc_option, u_old, tau)
    return u_new

# Неявная и Кранка–Николсона
def implicit_like_step(u_old, u_new, N, h, tau, t_k, t_k1, theta, bc_option):
    # Общая реализация θ-схемы: θ=1 неявная, θ=0.5 Кранка–Николсона
    x = np.linspace(0, L, N + 1)
    sigma = a_coef * tau / (h * h)
    M = N - 1 # число внутренних узлов

    # Коэффициенты трёхдиагональной системы
    a = np.zeros(M)
    b = np.zeros(M)
    c = np.zeros(M)
    d = np.zeros(M)

    # Формирование системы уравнений для внутренних точек
    for j in range(1, N):
        idx = j - 1
        a[idx] = 0.0 if idx == 0 else -theta * sigma
        b[idx] = 1.0 + 2.0 * theta * sigma
        c[idx] = -theta * sigma
        lap_k = (u_old[j + 1] - 2 * u_old[j] + u_old[j - 1]) / (h * h)
        rhs_f = (1 - theta) * f(x[j], t_k) + theta * f(x[j], t_k1)
        d[idx] = u_old[j] + a_coef * ((1 - theta) * tau * lap_k) + tau * rhs_f

    # Учитываем левое граничное условие через поток u_x(0,t)
    gk1_left = g_left(t_k1)
    d[0] += theta * sigma * (u_old[1] - h * gk1_left)

    # Учитываем правое граничное условие в зависимости от варианта аппроксимации
    gk1 = g_right(t_k1)
    if bc_option == 1:
        alpha, beta, gamma = 1.0, 0.0, h * gk1
    elif bc_option == 2:
        alpha, beta, gamma = 4.0 / 3.0, -1.0 / 3.0, (2 * h * gk1) / 3.0
    else:
        fNk1 = f(L, t_k1)
        denom = 1 + (h * h) / (2 * tau)
        alpha = 1 / denom
        gamma = (h * gk1 + (h * h / (2 * tau)) * u_old[N] + (h * h / 2) * fNk1) / denom
        beta = 0.0

    # Модифицируем последнюю строку матрицы для правой границы
    last = M - 1
    a[last] += c[last] * beta
    b[last] += c[last] * alpha
    d[last] -= c[last] * gamma
    c[last] = 0.0 # обнуляем, так как правая граница исключена

    # Решаем СЛАУ методом Томаса
    sol = thomas_solve(a, b, c, d)

    # Восстанавливаем полное решение с учётом граничных значений
    u_new[0] = u_new[1] - h * g_left(t_k1)
    u_new[1:N] = sol
    u_new[N] = apply_boundary_condition(u_new, N, h, t_k1, bc_option, u_old, tau)
    return u_new

# Основной цикл моделирования
def run_simulation(N, K, T, scheme, bc_option):
    # Определяем параметр θ в зависимости от схемы:
    # 0 — явная, 1 — неявная, 0.5 — Кранка–Николсона
    theta = 0.0 if scheme == 0 else (1.0 if scheme == 1 else 0.5)

    # Определяем шаг сетки и вычсиляем сигму - параметр сетки
    h = L / N
    tau = T / K
    sigma = a_coef * tau / (h * h)

    print(f"Parameters: N={N}, K={K}, T={T}, h={h}, tau={tau}")
    print(f"Scheme θ={theta} (0-explicit,1-implicit,0.5-CN), BC option={bc_option}")
    # Если шаг по времени слишком большой, решение становится неустойчивым
    if scheme == 0 and sigma > 0.5:
        print(f"Warning: explicit scheme may be unstable: sigma = {sigma:.3f} > 0.5")
    
    # Создаём сетку по x и задаём начальные условия u(x,0) = sin(x)
    x = np.linspace(0, L, N + 1)
    u_old = np.array([math.sin(xi) for xi in x]) # Решение на текущем временном шаге
    u_new = np.zeros(N + 1) # решение на следующем временном шаге

    # Основной временной цикл
    for k in range(K):
        t_k, t_k1 = k * tau, (k + 1) * tau
        if scheme == 0:
            u_new = explicit_step(u_old, u_new, N, h, tau, t_k, t_k1, bc_option)
        else:
            u_new = implicit_like_step(u_old, u_new, N, h, tau, t_k, t_k1, theta, bc_option)
        u_old[:] = u_new # переходим на следующий временной слой

    # Сравниваем с аналитическим решением
    exact_vals = np.array([exact(xj, T) for xj in x])
    errors = np.abs(u_old - exact_vals)
    max_err = np.max(errors)
    l2_err = math.sqrt(np.mean(errors ** 2))

    # Для BC3 строим график решений
    if bc_option == 3:
        scheme_name = ["Explicit", "Implicit", "Crank-Nicolson"][scheme]
        create_chart(x, u_old, exact_vals, scheme_name, T)

    print(f"Result at t = {T:.3f}: max error = {max_err:.3e}, L2 error = {l2_err:.3e}")
    return max_err, l2_err

def main():
    N, K, T = 100, 20000, 1.0 # Сетка и время моделирования
    max_errors, l2_errors, labels = [], [], []

    # Перебираем все схемы (явная, неявная, Кранк–Николсон)
    for scheme in range(3):
        # и все три варианта аппроксимации правой границы
        for bc_option in range(1, 4):
            print("=================================================")
            scheme_name = ["Explicit", "Implicit", "Crank-Nicolson"][scheme]
            label = f"{scheme_name}_BC{bc_option}"
            print(f"Scheme: {scheme_name}, BC option: {bc_option}")

            max_err, l2_err = run_simulation(N, K, T, scheme, bc_option)
            max_errors.append(max_err)
            l2_errors.append(l2_err)
            labels.append(label)

    # График сравнения ошибок
    create_error_chart(max_errors, l2_errors, labels)

if __name__ == "__main__":
    main()