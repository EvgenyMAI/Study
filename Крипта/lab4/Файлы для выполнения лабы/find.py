from sympy import isprime, mod_inverse, sqrt_mod
import random

# Проверка: y^2 = x^3 + ax + b mod p
def is_on_curve(x, y, a, b, p):
    return (y * y - (x ** 3 + a * x + b)) % p == 0

# Проверка на невырожденность: 4a^3 + 27b^2 != 0 mod p
def is_valid_curve(a, b, p):
    return (4 * a ** 3 + 27 * b ** 2) % p != 0

# Поиск точки на кривой: ищем x, для которого существует y такое, что y^2 = x^3 + ax + b mod p
def find_point_on_curve(a, b, p):
    for _ in range(1000000):
        x = random.randint(0, p - 1)
        rhs = (x**3 + a * x + b) % p
        y_vals = sqrt_mod(rhs, p, all_roots=True)
        if y_vals:
            return x, y_vals[0]
    return None

# Попробуем подобрать подходящую кривую и точку
def generate_large_curve_and_point(min_p=31_000_000, max_p=32_000_000):
    for p in range(min_p, max_p):
        if not isprime(p):
            continue
        for _ in range(10):  # Пробуем несколько a, b
            a = random.randint(0, p - 1)
            b = random.randint(0, p - 1)
            if not is_valid_curve(a, b, p):
                continue
            point = find_point_on_curve(a, b, p)
            if point:
                print({"p": p, "a": a, "b": b, "P": point})
                return {"p": p, "a": a, "b": b, "P": point}
    return None

generate_large_curve_and_point()