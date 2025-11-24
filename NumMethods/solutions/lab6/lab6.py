import numpy as np
import matplotlib.pyplot as plt

# Параметры задачи
L = 1.0   
T1 = 0.5  
T2 = 1.0  
N = 800   
K1 = 800  
K2 = 1600 

h = L / N 

# Аналитическое решение
def exact_solution(x, t):
    return np.exp(2*x) * np.cos(t)

# Начальное условие
def initial_shape(x):
    return np.exp(2*x)

def thomas(a, b, c, d):
    n = len(d)
    cp = np.zeros(n)
    dp = np.zeros(n)
    u = np.zeros(n)
    
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    
    for i in range(1, n):
        denom = b[i] - a[i] * cp[i-1]
        cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i] * dp[i-1]) / denom
    
    u[-1] = dp[-1]
    for i in range(n-2, -1, -1):
        u[i] = dp[i] - cp[i] * u[i+1]
    
    return u

def solve_explicit(T, K, approx_order):
    # Явная схема
    tau = T / K  
    sigma = tau**2 / h**2

    if sigma > 1:
        print(f"WARNING: нарушено условие устойчивости явной схемы: sigma={sigma:.3f} > 1")

    # Создаем сетку и массив решения
    x = np.linspace(0, L, N+1)
    t = np.linspace(0, T, K+1)
    u = np.zeros((K+1, N+1)) 
    
    # Задаём первое начальное условие
    u[0, :] = initial_shape(x) # e^{2x}
    
    # Аппроксимация второго начального условия
    if approx_order == 1:
        # Первый порядок
        u[1, :] = u[0, :]
    else:
        # Второй порядок через ряд Тейлора
        u_xx = np.zeros(N+1)
        for i in range(1, N):
            u_xx[i] = (u[0, i+1] - 2*u[0, i] + u[0, i-1]) / h**2
        
        # Граничные точки
        u_xx[0] = (u[0, 2] - 2*u[0, 1] + u[0, 0]) / h**2
        u_xx[N] = (u[0, N] - 2*u[0, N-1] + u[0, N-2]) / h**2
        
        u[1, :] = u[0, :] + 0.5 * tau**2 * (u_xx - 5*u[0, :])
    
    # Временные шаги
    for k in range(1, K):
        for i in range(1, N):
            u[k+1, i] = (2*u[k, i] - u[k-1, i] + sigma * (u[k, i+1] - 2*u[k, i] + u[k, i-1]) - 5 * tau**2 * u[k, i])
        
        # Граничные условия (двухточечная аппроксимация первого порядка точности)
        # Левое граничное условие: u_x(0,t) - 2u(0,t) = 0
        u[k+1, 0] = u[k+1, 1] / (1 + 2*h)
        # Правое граничное условие: u_x(1,t) - 2u(1,t) = 0  
        u[k+1, N] = u[k+1, N-1] / (1 - 2*h)
    
    return x, t, u

def solve_implicit(T, K, approx_order):
    # Неявная схема
    tau = T / K
    sigma = tau**2 / h**2
    
    # Создаем сетку и массив решения
    x = np.linspace(0, L, N+1)
    t = np.linspace(0, T, K+1)
    u = np.zeros((K+1, N+1))
    
    # Задаём первое начальное условие
    u[0, :] = initial_shape(x)
    
    # Аппроксимация второго начального условия
    if approx_order == 1:
        # Первый порядок
        u[1, :] = u[0, :]
    else:
        # Второй порядок
        u_xx = np.zeros(N+1)
        for i in range(1, N):
            u_xx[i] = (u[0, i+1] - 2*u[0, i] + u[0, i-1]) / h**2

        # Граничные точки
        u_xx[0] = (u[0, 2] - 2*u[0, 1] + u[0, 0]) / h**2
        u_xx[N] = (u[0, N] - 2*u[0, N-1] + u[0, N-2]) / h**2
        
        u[1, :] = u[0, :] + 0.5 * tau**2 * (u_xx - 5*u[0, :])
    
    # Временные шаги
    for k in range(1, K):
        a = np.zeros(N+1)  
        b = np.zeros(N+1)  
        c = np.zeros(N+1)  
        d = np.zeros(N+1)  
        
        # Левое граничное условие
        # (u1 - u0) / h - 2u0 => a0u0 + b0u1 + c0u2 = d0
        a[0] = 0.0
        b[0] = -1/h - 2
        c[0] = 1/h
        d[0] = 0.0
        
        for i in range(1, N):
            # Коэффициенты неявной схемы
            a[i] = -sigma
            b[i] = 1 + 5*tau**2 + 2*sigma
            c[i] = -sigma
            d[i] = 2*u[k, i] - u[k-1, i]
        
        # Правое граничное условие
        a[N] = -1/h
        b[N] = 1/h - 2
        c[N] = 0.0
        d[N] = 0.0
        
        u[k+1, :] = thomas(a, b, c, d)
    
    return x, t, u

print("="*60)
print("ЧИСЛЕННОЕ РЕШЕНИЕ ГИПЕРБОЛИЧЕСКОГО УРАВНЕНИЯ")
print("="*60)

print("Расчет для T = 0.5...")
x1, t1, u_explicit_05_1 = solve_explicit(T1, K1, 1)
x1, t1, u_implicit_05_1 = solve_implicit(T1, K1, 1)
x1, t1, u_explicit_05_2 = solve_explicit(T1, K1, 2)
x1, t1, u_implicit_05_2 = solve_implicit(T1, K1, 2)

print("Расчет для T = 1.0...")
x2, t2, u_explicit_10_1 = solve_explicit(T2, K2, 1)
x2, t2, u_implicit_10_1 = solve_implicit(T2, K2, 1)
x2, t2, u_explicit_10_2 = solve_explicit(T2, K2, 2)
x2, t2, u_implicit_10_2 = solve_implicit(T2, K2, 2)

u_analytical_05 = exact_solution(x1, T1)
u_analytical_10 = exact_solution(x2, T2)

error_explicit_05_1 = np.abs(u_explicit_05_1[-1, :] - u_analytical_05)
error_implicit_05_1 = np.abs(u_implicit_05_1[-1, :] - u_analytical_05)
error_explicit_05_2 = np.abs(u_explicit_05_2[-1, :] - u_analytical_05)
error_implicit_05_2 = np.abs(u_implicit_05_2[-1, :] - u_analytical_05)

error_explicit_10_1 = np.abs(u_explicit_10_1[-1, :] - u_analytical_10)
error_implicit_10_1 = np.abs(u_implicit_10_1[-1, :] - u_analytical_10)
error_explicit_10_2 = np.abs(u_explicit_10_2[-1, :] - u_analytical_10)
error_implicit_10_2 = np.abs(u_implicit_10_2[-1, :] - u_analytical_10)

print("\n" + "="*50)
print("МАКСИМАЛЬНЫЕ ПОГРЕШНОСТИ")
print("="*50)
print(f"T = 0.5:")
print(f"  Явная схема (1 порядок): {np.max(error_explicit_05_1):.6f}")
print(f"  Неявная схема (1 порядок): {np.max(error_implicit_05_1):.6f}")
print(f"  Явная схема (2 порядок): {np.max(error_explicit_05_2):.6f}")
print(f"  Неявная схема (2 порядок): {np.max(error_implicit_05_2):.6f}")
print(f"T = 1.0:")
print(f"  Явная схема (1 порядок): {np.max(error_explicit_10_1):.6f}")
print(f"  Неявная схема (1 порядок): {np.max(error_implicit_10_1):.6f}")
print(f"  Явная схема (2 порядок): {np.max(error_explicit_10_2):.6f}")
print(f"  Неявная схема (2 порядок): {np.max(error_implicit_10_2):.6f}")

plt.figure(figsize=(20, 12))

plt.subplot(2, 3, 1)
plt.plot(x1, u_analytical_05, 'k-', linewidth=2, label='Аналитическое')
plt.plot(x1, u_explicit_05_1[-1, :], 'r--', linewidth=1.5, label='Явная схема')
plt.plot(x1, u_implicit_05_1[-1, :], 'b:', linewidth=1.5, label='Неявная схема')
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.title('Решение при t=0.5 (1 порядок)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 2)
plt.plot(x1, u_analytical_05, 'k-', linewidth=2, label='Аналитическое')
plt.plot(x1, u_explicit_05_2[-1, :], 'r--', linewidth=1.5, label='Явная схема')
plt.plot(x1, u_implicit_05_2[-1, :], 'b:', linewidth=1.5, label='Неявная схема')
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.title('Решение при t=0.5 (2 порядок)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 3)
plt.plot(x2, u_analytical_10, 'k-', linewidth=2, label='Аналитическое')
plt.plot(x2, u_explicit_10_1[-1, :], 'r--', linewidth=1.5, label='Явная схема')
plt.plot(x2, u_implicit_10_1[-1, :], 'b:', linewidth=1.5, label='Неявная схема')
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.title('Решение при t=1.0 (1 порядок)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 4)
plt.plot(x2, u_analytical_10, 'k-', linewidth=2, label='Аналитическое')
plt.plot(x2, u_explicit_10_2[-1, :], 'r--', linewidth=1.5, label='Явная схема')
plt.plot(x2, u_implicit_10_2[-1, :], 'b:', linewidth=1.5, label='Неявная схема')
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.title('Решение при t=1.0 (2 порядок)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 5)
plt.semilogy(x1, error_explicit_05_1, 'r--', linewidth=1, label='Явная (1 порядок)')
plt.semilogy(x1, error_implicit_05_1, 'r:', linewidth=1, label='Неявная (1 порядок)')
plt.semilogy(x1, error_explicit_05_2, 'b--', linewidth=1, label='Явная (2 порядок)')
plt.semilogy(x1, error_implicit_05_2, 'b:', linewidth=1, label='Неявная (2 порядок)')
plt.xlabel('x')
plt.ylabel('Погрешность')
plt.title('Погрешности при t=0.5')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 6)
plt.semilogy(x2, error_explicit_10_1, 'r--', linewidth=1, label='Явная (1 порядок)')
plt.semilogy(x2, error_implicit_10_1, 'r:', linewidth=1, label='Неявная (1 порядок)')
plt.semilogy(x2, error_explicit_10_2, 'b--', linewidth=1, label='Явная (2 порядок)')
plt.semilogy(x2, error_implicit_10_2, 'b:', linewidth=1, label='Неявная (2 порядок)')
plt.xlabel('x')
plt.ylabel('Погрешность')
plt.title('Погрешности при t=1.0')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nРасчет завершен!")