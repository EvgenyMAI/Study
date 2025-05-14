import matplotlib.pyplot as plt

def compute_derivatives(x, y, x_star):
    i = -1
    for j in range(len(x) - 1):
        if x[j] <= x_star <= x[j + 1]:
            i = j + 1
            break

    if i == -1 or i + 1 >= len(x):
        print("Ошибка: x* вне диапазона данных или недостаточно точек.")
        return

    # Первая производная (левосторонняя и правосторонняя)
    left_diff = (y[i] - y[i - 1]) / (x[i] - x[i - 1])
    right_diff = (y[i + 1] - y[i]) / (x[i + 1] - x[i])

    # Формулы 3.20 (приближение первой производной в x*)
    temp = (right_diff - left_diff) / (x[i + 1] - x[i - 1])
    first_derivative = left_diff + temp * (2 * x_star - x[i] - x[i - 1])

    # Формула 3.21 (вторая производная в x*)
    second_derivative = 2 * temp

    print(f"Левосторонняя производная: {left_diff:.10f}")
    print(f"Правосторонняя производная: {right_diff:.10f}")
    print(f"Первая производная в x*={x_star}: {first_derivative:.10f}")
    print(f"Вторая производная в x*={x_star}: {second_derivative:.10f}")

    # Визуализация
    plt.figure(figsize=(10, 6))

    # График точек
    plt.plot(x, y, 'bo-', label='Табличные значения')

    # Левая секущая
    x_left = [x[i - 1], x[i]]
    y_left = [y[i - 1], y[i]]
    plt.plot(x_left, y_left, 'g--', label='Левосторонняя секущая')

    # Правая секущая
    x_right = [x[i], x[i + 1]]
    y_right = [y[i], y[i + 1]]
    plt.plot(x_right, y_right, 'r--', label='Правосторонняя секущая')

    # Приближённая касательная (в x*)
    x_tangent = [x_star - 0.05, x_star + 0.05]
    y_tangent = [y[i] + first_derivative * (xi - x[i]) for xi in x_tangent]
    plt.plot(x_tangent, y_tangent, 'm--', label='Касательная (1-я производная)')

    # Вертикальная линия на x*
    plt.axvline(x_star, color='k', linestyle=':', label='x*')

    # Настройки
    plt.title("Приближение производных по табличным данным")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()

# Пример данных
x = [0, 0.1, 0.2, 0.3, 0.4]
y = [1, 1.1052, 1.2214, 1.3499, 1.4918]
x_star = 0.2

compute_derivatives(x, y, x_star)