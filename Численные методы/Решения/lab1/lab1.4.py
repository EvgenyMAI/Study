import math

def is_symmetric(matrix):
    """Проверяет, является ли матрица симметричной."""
    n = len(matrix)
    for i in range(n):
        for j in range(n):
            if matrix[i][j] != matrix[j][i]:
                return False
    return True

def find_max_off_diagonal(matrix):
    """Находит индекс максимального по модулю внедиагонального элемента."""
    n = len(matrix)
    max_value = 0
    i_max, j_max = 0, 1
    for i in range(n):
        for j in range(i + 1, n):
            if abs(matrix[i][j]) > abs(max_value):
                max_value = matrix[i][j]
                i_max, j_max = i, j
    return i_max, j_max, max_value

def sum_off_diagonal_squares(matrix):
    """Вычисляет сумму квадратов внедиагональных элементов матрицы."""
    n = len(matrix)
    total = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                total += matrix[i][j] ** 2
    return total

def create_rotation_matrix(n, i_max, j_max, phi):
    """Создает матрицу поворота U."""
    U = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    U[i_max][i_max] = math.cos(phi)
    U[j_max][j_max] = math.cos(phi)
    U[i_max][j_max] = -math.sin(phi)
    U[j_max][i_max] = math.sin(phi)
    return U

def multiply_matrices(A, B):
    """Перемножает две матрицы."""
    rows, cols = len(A), len(B[0])
    common_dim = len(B)
    result = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            for k in range(common_dim):
                result[i][j] += A[i][k] * B[k][j]
    return result

def transpose_matrix(matrix):
    """Транспонирует матрицу."""
    n, m = len(matrix), len(matrix[0])
    return [[matrix[j][i] for j in range(n)] for i in range(m)]

def jacobi_eigenvalues_and_vectors(matrix, epsilon=1e-5, max_iterations=100):
    """
    Вычисляет собственные значения и собственные векторы симметрической матрицы методом Якоби.
    """
    if not is_symmetric(matrix):
        raise ValueError("Матрица не является симметричной")
    
    n = len(matrix)
    eigenvectors = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    iterations = 0
    
    while True:
        if sum_off_diagonal_squares(matrix) < epsilon or iterations >= max_iterations:
            break
        
        i_max, j_max, max_off_diag = find_max_off_diagonal(matrix)
        iterations += 1
        phi = 0.5 * math.atan(2 * max_off_diag / (matrix[i_max][i_max] - matrix[j_max][j_max]))
        
        U = create_rotation_matrix(n, i_max, j_max, phi)
        matrix = multiply_matrices(multiply_matrices(transpose_matrix(U), matrix), U) # После этого внедиагональный элемент a[ij] стремится к нулю
        eigenvectors = multiply_matrices(eigenvectors, U) # Обновление матрицы собственных векторов
    
    eigenvalues = [matrix[i][i] for i in range(n)]
    return eigenvalues, eigenvectors

matrix = [[-3, -1, 3],
          [-1, 8, 1],
          [3, 1, 5]]

eigenvalues, eigenvectors = jacobi_eigenvalues_and_vectors(matrix)

print("Собственные значения:")
for i, eigenvalue in enumerate(eigenvalues):
    print(f'λ_{i} = {eigenvalue:.6f}')

print("\nСобственные векторы:")
for i, eigenvector in enumerate(eigenvectors):
    print(f'v_{i} = {eigenvector}')