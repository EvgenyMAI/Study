import time
import platform
from sympy import mod_inverse
from cpuinfo import get_cpu_info

# Нулевая точка
O = None

# Сложение точек на кривой
def elliptic_add(P, Q, a, p):
    if P == O:
        return Q
    if Q == O:
        return P

    x1, y1 = P
    x2, y2 = Q

    if x1 == x2 and (y1 + y2) % p == 0:
        return O

    if P == Q:
        m = (3 * x1 * x1 + a) * mod_inverse(2 * y1, p) % p
    else:
        m = (y2 - y1) * mod_inverse((x2 - x1) % p, p) % p

    x3 = (m * m - x1 - x2) % p
    y3 = (m * (x1 - x3) - y1) % p
    return (x3, y3)

# Умножение точки на скаляр
def elliptic_mul(k, P, a, p):
    result = O
    addend = P

    while k:
        if k & 1:
            result = elliptic_add(result, addend, a, p)
        addend = elliptic_add(addend, addend, a, p)
        k >>= 1
    return result

# Полный перебор порядка
def find_order(P, a, p, max_iter):
    for i in range(1, max_iter + 1):
        R = elliptic_mul(i, P, a, p)
        if R == O:
            return i
    return None

def print_system_info():
    print("СИСТЕМНАЯ ИНФОРМАЦИЯ")
    print("ОС:", platform.system(), platform.release())
    print("Python:", platform.python_version())
    info = get_cpu_info()
    print("CPU:", info["brand_raw"])
    print("Ядер:", info["count"])
    print()

if __name__ == "__main__":
    print_system_info()

    # Параметры кривой
    p = 31000003
    a = 11074508
    b = 22681662
    P = (5679991, 14430394)

    print(f"Кривая: y² = x³ + {a}x + {b} mod {p}")
    print(f"Точка: P = {P}")
    print("Начинается полный перебор порядка точки...\n")

    start = time.time()
    order = find_order(P, a, p, 18_000_000)
    end = time.time()

    if order:
        print(f"Порядок точки: {order}")
    else:
        print("Порядок не найден в пределах max_iter")

    print(f"\nВремя выполнения: {((end - start) / 60):.2f} минут")