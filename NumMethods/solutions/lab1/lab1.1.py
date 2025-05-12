import numpy as np

def find_main_element(matrix, row):
    '''
    Ищет индекс главного элемента в указанном столбце
    '''
    main_string_index = row
    main_number = abs(matrix[row, row])

    for string in range(row + 1, len(matrix)):
        if abs(matrix[string, row]) > main_number:
            main_number = abs(matrix[string, row])
            main_string_index = string
    return main_string_index

def lu_decompose(matrix):
    '''
    Вычисляет LU-разложение
    '''
    n = len(matrix)
    L = np.zeros((n, n))
    U = matrix.copy()
    P = np.arange(n) # Массив перестановок строк (по умолчанию [0, 1, 2, ...])
    swaps = 0

    for column in range(n):
        string_with_max_element = find_main_element(U, column)
        # Если главный элемент не на диагонали
        if string_with_max_element != column:
            # Меняем строки местами
            U[[column, string_with_max_element]] = U[[string_with_max_element, column]]
            P[column], P[string_with_max_element] = P[string_with_max_element], P[column]
            L[[column, string_with_max_element]] = L[[string_with_max_element, column]]
            swaps += 1

        for string in range(column + 1, n):
            nulling_coefficient = U[string, column] / U[column, column]
            L[string, column] = nulling_coefficient
            U[string, column:] -= nulling_coefficient * U[column, column:]

    np.fill_diagonal(L, 1.0)
    return L, U, P, swaps

def determinant(U, swaps):
    '''
    Вычисляет определитель
    '''
    det = np.prod(np.diag(U)) # Произведение диагональных элементов
    return det if swaps % 2 == 0 else -det

def solve_system(L, U, P, b):
    '''
    Решает систему уравнений LUx = b
    '''
    n = len(L)
    b_permuted = b[P] # Переставляем b в новом порядке

    # Решение y[i] + sum(Ly) = Pb (прямой ход) - на диагонали единицы
    y = np.zeros(n)
    for i in range(n):
        # np.dot() - сумма произведений элементов L и уже найденных y
        y[i] = b_permuted[i] - np.dot(L[i, :i], y[:i])

    # Решение Ux = y (обратный ход)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        # np.dot() - сумма произведений элементов U и уже найденных x
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]

    return x

def compute_inverse_matrix(L, U, P):
    '''
    Вычисляет обратную матрицу
    '''
    n = len(L)
    inv = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1
        inv[:, i] = solve_system(L, U, P, e) # Решаем систему уравнений LUx = e[i]
    return inv

# --------------------------------------------

A = np.array([
    [-9, 8, 8, 6],
    [-7, -9, 5, 4],
    [-3, -1, 8, 0],
    [3, -1, -4, -5]
], dtype=float)

b = np.array([-81, -50, -69, 48], dtype=float)

L, U, P, swaps = lu_decompose(A)

det = determinant(U, swaps)
print(f"Детерминант: {int(round(det))}")

solution = solve_system(L, U, P, b)
print("Решение системы:", " ".join(map(str, map(int, solution))))

A_inv = compute_inverse_matrix(L, U, P)
print("Обратная матрица:")
for row in A_inv:
    print(" ".join(f"{val:.6f}" for val in row))