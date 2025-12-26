import numpy as np
import math
import time
import matplotlib.pyplot as plt

# ЗАДАНИЕ РАСЧЁТНОЙ СЕТКИ

# Область решения:
# x ∈ [0, π], y ∈ [0, 1]
# Nx, Ny — количество разбиений (интервалов) по x и y

Nx = 63  # число интервалов по x
Ny = 40  # число интервалов по y

x0, x1 = 0.0, math.pi
y0, y1 = 0.0, 1.0

# Узлы сетки
x = np.linspace(x0, x1, Nx+1)
y = np.linspace(y0, y1, Ny+1)

# Шаги сетки
hx = x[1] - x[0]
hy = y[1] - y[0]

# АНАЛИТИЧЕСКОЕ РЕШЕНИЕ

# U(x,y) = sin(x) * e^y
def U_exact(X, Y):
    return np.sin(X) * np.exp(Y)

# ГРАНИЧНЫЕ УСЛОВИЯ НЕЙМАНА ПО x

# Слева (x = 0): u_x(0, y) = e^y
# Справа (x = π): u_x(π, y) = -e^y
# знак учитывается отдельно в коде
def g_neumann(yv):
    return np.exp(yv)

# НАЧАЛЬНОЕ ПРИБЛИЖЕНИЕ

# Для итерационных методов необходимо стартовое приближение.
# Используется линейная интерполяция по y между: u(x,0) = sin(x), u(x,1) = e * sin(x)
def build_grid():
    X, Y = np.meshgrid(x, y, indexing='ij')
    u = np.zeros_like(X)

    # Граничные условия Дирихле по y
    u[:,0]  = np.sin(x)          # y = 0
    u[:,-1] = math.e * np.sin(x) # y = 1

    # Линейная интерполяция по y
    for i in range(Nx+1):
        u[i,:] = np.linspace(u[i,0], u[i,-1], Ny+1)

    return u

# МЕТОД ЛИБМАНА

# Все значения нового слоя вычисляются только на основе значений предыдущей итерации.
# Используется центрально-разностная аппроксимация (второго порядка)
def jacobi_step(u):
    ax = hy*hy
    ay = hx*hx
    denom = 2*(hx*hx + hy*hy)

    # Создаём новый массив
    u_new = u.copy()

    # внутренние узлы сетки
    u_new[1:-1,1:-1] = (
        ax*(u[2:,1:-1] + u[:-2,1:-1]) +
        ay*(u[1:-1,2:] + u[1:-1,:-2])
    ) / denom

    # ЛЕВАЯ ГРАНИЦА x = 0 (Нейман)
    # Используется метод фиктивной точки:
    # u_{-1,j} = u_{1,j} - 2*h_x*u_x(0,y)
    j_idx = np.arange(1, Ny)
    g = g_neumann(y[j_idx])

    coeff = -2.0/(hx*hx) - 2.0/(hy*hy)
    other = (
        2*u[1,1:-1]/(hx*hx) +
        (u[0,2:] + u[0,:-2])/(hy*hy) -
        2.0*g/hx
    )

    u_new[0,1:-1] = - other / coeff

    # ПРАВАЯ ГРАНИЦА x = Pi (Нейман)
    # u_x(Pi,y) = -e^y
    g_r = g_neumann(y[1:Ny])
    gn = -g_r

    other_r = (
        2*u[-2,1:-1]/(hx*hx) +
        (u[-1,2:] + u[-1,:-2])/(hy*hy) +
        2.0*gn/hx
    )

    u_new[-1,1:-1] = - other_r / coeff

    # условие Дирихле по y
    u_new[:,0]  = np.sin(x)
    u_new[:,-1] = math.e * np.sin(x)

    return u_new

# МЕТОД ГАУССА–ЗЕЙДЕЛЯ

# Отличие от Якоби:
# - значения обновляются сразу на месте
# - метод сходится быстрее
def gauss_seidel_step(u):
    ax = hy*hy
    ay = hx*hx
    denom = 2*(hx*hx + hy*hy)

    # Внутренние узлы
    for i in range(1, Nx):
        for j in range(1, Ny):
            u[i,j] = (
                ax*(u[i+1,j] + u[i-1,j]) +
                ay*(u[i,j+1] + u[i,j-1])
            ) / denom

    # Левая граница (Нейман)
    for j in range(1, Ny):
        g = g_neumann(y[j])
        coeff = -2.0/(hx*hx) - 2.0/(hy*hy)
        other = (
            2*u[1,j]/(hx*hx) +
            (u[0,j+1] + u[0,j-1])/(hy*hy) -
            2.0*g/hx
        )
        u[0,j] = - other / coeff

    # Правая граница (Нейман)
    for j in range(1, Ny):
        g = g_neumann(y[j])
        gn = -g
        coeff = -2.0/(hx*hx) - 2.0/(hy*hy)
        other = (
            2*u[Nx-1,j]/(hx*hx) +
            (u[Nx,j+1] + u[Nx,j-1])/(hy*hy) +
            2.0*gn/hx
        )
        u[Nx,j] = - other / coeff

    # Условия Дирихле по y
    u[:,0]  = np.sin(x)
    u[:,-1] = math.e * np.sin(x)

    return u

# МЕТОД SOR (СВЕРХРЕЛАКСАЦИЯ)

# Улучшение метода Зейделя:  u_new = (1 - omega) * u_old + omega * u_GS
# При omega = 1 -> метод Зейделя
# При 1 < omega < 2 -> ускорение сходимости
def sor_step(u, omega):
    ax = hy*hy
    ay = hx*hx
    denom = 2*(hx*hx + hy*hy)

    # Внутренние узлы
    for i in range(1, Nx):
        for j in range(1, Ny):
            new = (
                ax*(u[i+1,j] + u[i-1,j]) +
                ay*(u[i,j+1] + u[i,j-1])
            ) / denom
            u[i,j] = (1.0 - omega) * u[i,j] + omega * new

    # Левая граница
    for j in range(1, Ny):
        g = g_neumann(y[j])
        coeff = -2.0/(hx*hx) - 2.0/(hy*hy)
        other = (
            2*u[1,j]/(hx*hx) +
            (u[0,j+1] + u[0,j-1])/(hy*hy) -
            2.0*g/hx
        )
        new = - other / coeff
        u[0,j] = (1.0 - omega) * u[0,j] + omega * new

    # Правая граница
    for j in range(1, Ny):
        g = g_neumann(y[j])
        gn = -g
        coeff = -2.0/(hx*hx) - 2.0/(hy*hy)
        other = (
            2*u[Nx-1,j]/(hx*hx) +
            (u[Nx,j+1] + u[Nx,j-1])/(hy*hy) +
            2.0*gn/hx
        )
        new = - other / coeff
        u[Nx,j] = (1.0 - omega) * u[Nx,j] + omega * new

    # Условия Дирихле по y
    u[:,0]  = np.sin(x)
    u[:,-1] = math.e * np.sin(x)

    return u

# ОБЁРТКА ИТЕРАЦИЙ

# Выполняет итерации до достижения точности tol или превышения максимального числа итераций
def iterate(method, tol=1e-6, max_iter=20000, omega=1.7, verbose=False):
    u = build_grid()
    it = 0

    if method == 'jacobi':
        while True:
            u_new = jacobi_step(u)
            diff = np.max(np.abs(u_new - u))
            u[:] = u_new[:]
            it += 1
            if diff < tol or it >= max_iter:
                break
        return u, it, diff

    elif method == 'gs':
        while True:
            u_old = u.copy()
            gauss_seidel_step(u)
            diff = np.max(np.abs(u - u_old))
            it += 1
            if diff < tol or it >= max_iter:
                break
        return u, it, diff

    elif method == 'sor':
        while True:
            u_old = u.copy()
            sor_step(u, omega)
            diff = np.max(np.abs(u - u_old))
            it += 1
            if diff < tol or it >= max_iter:
                break
        return u, it, diff

    else:
        raise ValueError("Unknown method")

# ЗАПУСК ВЫЧИСЛЕНИЙ

tol = 1e-6

# Оценка оптимального параметра релаксации
omega_est = 2.0 / (1.0 + math.sin(math.pi / max(Nx, Ny)))
omega = min(max(omega_est, 1.0), 1.95)

print("Запуск вычислений...")

u_jac, it_j, d_j = iterate('jacobi', tol=tol, max_iter=20000)
print(f"Jacobi: итераций={it_j}, diff={d_j:.2e}")

u_gs, it_g, d_g = iterate('gs', tol=tol, max_iter=20000)
print(f"Gauss-Seidel: итераций={it_g}, diff={d_g:.2e}")

u_sor, it_s, d_s = iterate('sor', tol=tol, max_iter=20000, omega=omega)
print(f"SOR (omega={omega:.3f}): итераций={it_s}, diff={d_s:.2e}")

# ВЫЧИСЛЕНИЕ ПОГРЕШНОСТЕЙ

# точное решение
X, Y = np.meshgrid(x, y, indexing='ij')
Ue = U_exact(X, Y)

# индексы сечений
y_slice1 = 0.5
y_slice2 = 1.0
j1 = int(round(y_slice1 / hy))
j2 = int(round(y_slice2 / hy))
print(f"Используем узлы сечений: y[{j1}] = {y[j1]:.6f}, y[{j2}] = {y[j2]:.6f}")

# Вычисление ошибок
err_j = np.abs(u_jac - Ue)
err_s = np.abs(u_gs - Ue)
err_o = np.abs(u_sor - Ue)

print(f"\nМакс. ошибки:")
print(f"Либман = {np.max(err_j):.3e}")
print(f"Зейдель = {np.max(err_s):.3e}")
print(f"SOR = {np.max(err_o):.3e}")

# ПОСТРОЕНИЕ ГРАФИКОВ

plt.figure(figsize=(12, 10))

# График 1: Сравнение решений при y = 0.5
plt.subplot(3, 1, 1)
plt.plot(x, Ue[:, j1], 'k--', linewidth=2, label='Точное')
plt.plot(x, u_jac[:, j1], 'b-', label='Либман')
plt.plot(x, u_gs[:, j1], 'g-', label='Зейдель')
plt.plot(x, u_sor[:, j1], 'm-', label=f'SOR (omega={omega:.2f})')
plt.title(f'Сравнение решений при y = {y[j1]:.3f}')
plt.xlabel('x')
plt.ylabel('u(x,y)')
plt.legend()
plt.grid(True)

# График 2: Сравнение решений при y = 1.0
plt.subplot(3, 1, 2)
plt.plot(x, Ue[:, j2], 'k--', linewidth=2, label='Точное')
plt.plot(x, u_jac[:, j2], 'b-', label='Либман')
plt.plot(x, u_gs[:, j2], 'g-', label='Зейдель')
plt.plot(x, u_sor[:, j2], 'm-', label=f'SOR (omega={omega:.2f})')
plt.title(f'Сравнение решений при y = {y[j2]:.3f}')
plt.xlabel('x')
plt.ylabel('u(x,y)')
plt.legend()
plt.grid(True)

# График 3: Сравнение ошибок методов
plt.subplot(3, 1, 3)
# Берем сечение по x = Pi/2 для сравнения ошибок по y
x_idx = Nx // 2  # индекс для x = Pi/2
plt.plot(y, err_j[x_idx, :], 'r-', label='Либман')
plt.plot(y, err_s[x_idx, :], 'g-', label='Зейдель')
plt.plot(y, err_o[x_idx, :], 'b-', label='SOR')
plt.title('Сравнение ошибок методов при x = π/2')
plt.xlabel('y')
plt.ylabel('|u_num - u_exact|')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()