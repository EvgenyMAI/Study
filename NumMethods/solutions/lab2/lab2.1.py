import math

EPSILON = 1e-6
MAX_ITERATIONS = 100

def f(x):
    return math.log10(x + 1) - x + 0.5

def df(x):
    return 1 / ((x + 1) * math.log(10)) - 1

def d2f(x):
    return -1 / ((x + 1) ** 2 * math.log(10))

def phi(x):
    return math.log10(x + 1) + 0.5

def phi_derivative_max(a, b):
    # φ'(x) = 1 / ((x + 1) * ln(10)) — максимум в a
    return 1 / ((a + 1) * math.log(10))

def find_root_interval(f, start, end, step):
    x_prev = start
    f_prev = f(x_prev)
    x = start + step
    while x <= end:
        f_curr = f(x)
        if f_prev * f_curr <= 0:
            return [x_prev, x]
        x_prev = x
        f_prev = f_curr
        x += step
    return None

def newton_method(f, df, d2f, x0, a, b, eps):
    print("\nМетод Ньютона:")

    # Условие сходимости f(x0)*f''(x0) > 0 
    check_value = f(x0) * d2f(x0)
    if check_value <= 0:
        print(f"Предупреждение: f(x0)*f''(x0) = {check_value:.2f} <= 0 (условие сходимости не выполняется)")

    x = x0
    print(f"iter {0:2d}: x = {x:.8f}")

    for iter in range(1, MAX_ITERATIONS + 1):
        fx = f(x)
        dfx = df(x)

        if abs(dfx) < 1e-12:
            print("Ошибка: производная слишком близка к нулю!")
            return

        x_next = x - fx / dfx

        if x_next < a or x_next > b:
            print("Ошибка: выход за границы интервала!")
            return

        delta = abs(x_next - x)
        print(f"iter {iter:2d}: x = {x_next:.8f} | Δ = {delta:.2e}")

        if delta < eps:
            print("Достигнута заданная точность")
            return

        x = x_next

    print("Достигнуто максимальное число итераций!")

def iteration_method(phi, x0, a, b, eps):
    print("\nМетод простой итерации:")

    # Условие сходимости сущ q такой что abs(phi'(x) <= q < 1) для всех x на интервале
    q = phi_derivative_max(a, b)
    if q >= 1:
        print(f"Ошибка: условие сходимости не выполняется (q = {q:.2f})")
        return
    print(f"Параметр сходимости q = {q:.6f}")

    x = x0
    print(f"iter {0:2d}: x = {x:.8f}")

    for iter in range(1, MAX_ITERATIONS + 1):
        x_next = phi(x)

        if x_next < a or x_next > b:
            print("Ошибка: выход за границы интервала!")
            return
        
        # Условие окончания abs(x* - x(k+1)) <= (q/(1-q)) * abs(x(k+1)-x(k))
        delta = abs(x_next - x)
        error_estimate = q * delta / (1 - q)
        print(f"iter {iter:2d}: x = {x_next:.8f} | Δ = {delta:.2e} | оценка погрешности = {error_estimate:.2e}")

        if error_estimate < eps:
            print("Достигнута заданная точность")
            return

        x = x_next

    print("Достигнуто максимальное число итераций!")

def main():
    print(f"Установлена точность ε = {EPSILON:.1e}")

    interval = find_root_interval(f, 0, 2, 0.01)
    if not interval:
        print("Не удалось найти интервал с корнем!")
        return

    a, b = interval
    print(f"Найден интервал с корнем: [{a:.4f}, {b:.4f}]")
    x0 = (a + b) / 2
    print(f"Начальное приближение: x0 = {x0:.6f}")

    newton_method(f, df, d2f, x0, a, b, EPSILON)
    iteration_method(phi, x0, a, b, EPSILON)

if __name__ == "__main__":
    main()