import numpy as np
import matplotlib.pyplot as plt

# === 1. Входные данные ===
points = [
    (1.0, 3.4142),
    (1.9, 2.9818),
    (2.8, 3.3095),
    (3.7, 3.8184),
    (4.6, 4.3599),
    (5.5, 4.8318)
]

x_vals = np.array([p[0] for p in points])
y_vals = np.array([p[1] for p in points])
n = len(points)

# === 2. Решение нормальных систем МНК ===

def solve_least_squares(x, y, degree):
    """Находит коэффициенты приближающего многочлена степени degree методом наименьших квадратов (МНК)."""
    # Составляем матрицу Вандермонда для подстановки соответствующих xi^j в полином
    A = np.vander(x, N=degree+1, increasing=True) # Строит матрицу, где каждая строка — это степенные комбинации соответствующего x_i (xi^0, ..., xi^n)
    # Решаем нормальную систему A.T A a = A.T y
    ATA = A.T @ A
    ATy = A.T @ y
    coeffs = np.linalg.solve(ATA, ATy)
    return coeffs # Коэффициенты многочлена, минимизирующие ошибку

# === 3. Вычисление ошибки ===

def compute_error(x, y, coeffs):
    """Считает сумму квадратов ошибок."""
    y_pred = np.polyval(coeffs[::-1], x) # Вычисляет значение многочлена на заданных x - вместо наивной подстановки точки в полином используется схема Горнера
    return np.sum((y - y_pred)**2)

# === 4. Построение графика ===

def plot_results(x, y, coeffs_list, labels):
    """Строит график исходных точек и аппроксимаций."""
    plt.figure(figsize=(10, 6))
    
    # Исходные точки
    plt.scatter(x, y, color='black', label='Исходные данные')
    
    x_plot = np.linspace(min(x) - 0.2, max(x) + 0.2, 500)
    
    colors = ['blue', 'red']
    for coeffs, label, color in zip(coeffs_list, labels, colors):
        y_plot = np.polyval(coeffs[::-1], x_plot)
        plt.plot(x_plot, y_plot, label=label, color=color)
    
    plt.title("Аппроксимация методом наименьших квадратов")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# === 5. Основная логика ===

# Линейная аппроксимация (1-я степень)
coeffs_linear = solve_least_squares(x_vals, y_vals, degree=1)
error_linear = compute_error(x_vals, y_vals, coeffs_linear)

# Квадратичная аппроксимация (2-я степень)
coeffs_quadratic = solve_least_squares(x_vals, y_vals, degree=2)
error_quadratic = compute_error(x_vals, y_vals, coeffs_quadratic)

# === 6. Вывод результатов ===

def print_polynomial(coeffs):
    terms = [f"{coeff:.6f}·x^{i}" for i, coeff in enumerate(coeffs)]
    print(" + ".join(terms))

print("Многочлен 1-й степени:")
print_polynomial(coeffs_linear)
print(f"Сумма квадратов ошибок: {error_linear:.6f}\n")

print("Многочлен 2-й степени:")
print_polynomial(coeffs_quadratic)
print(f"Сумма квадратов ошибок: {error_quadratic:.6f}\n")

# === 7. График ===
plot_results(x_vals, y_vals, [coeffs_linear, coeffs_quadratic], ["1-я степень", "2-я степень"])