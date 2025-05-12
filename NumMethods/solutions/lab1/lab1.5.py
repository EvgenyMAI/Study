import numpy as np

def euclidean_norm(a):
    """Вычисляет евклидову норму вектора a."""
    return np.sqrt(np.sum(a**2))

def householder_reflection(a, i, n):
    """Вычисляет полную матрицу Хаусхолдера для вектора a, начиная с позиции i."""
    v = np.zeros(n)
    v[i:] = a
    v[i] += np.sign(a[0]) * euclidean_norm(a)
    beta = np.dot(v, v)
    H = np.eye(n) - 2 * np.outer(v, v) / beta
    return H

def qr_decomposition(A):
    """Выполняет QR-разложение матрицы A с использованием преобразования Хаусхолдера."""
    n = A.shape[0]
    Q = np.eye(n)
    R = A.copy()
    
    for i in range(n - 1):
        # вектор a, содержит текущий диагональный элемент и все элементы ниже него
        a = R[i:, i]
        
        # Вычисляем полную матрицу Хаусхолдера
        H = householder_reflection(a, i, n)
        
        # Обновляем R и Q
        R = np.dot(H, R) # зануляет нужные поддиагональные элементы
        Q = np.dot(Q, H)

    return Q, R

def extract_eigenvalues(A, eps=1e-10, max_iterations=100):
    n = A.shape[0]
    eigenvalues = []
    i = 0
    while i < n:
        if i == n - 1 or np.abs(A[i+1, i]) < eps:
            # 1×1 блок (вещественное собственное значение)
            eigenvalues.append(A[i, i])
            i += 1
        else:
            # 2×2 блок (комплексно-сопряженная пара)
            block = A[i:i+2, i:i+2]
            lambda_prev = None
            converged = False

            for _ in range(max_iterations):
                current_eigvals = np.linalg.eigvals(block)

                if lambda_prev is not None:
                    if all(np.abs(current_eigvals - lambda_prev) < eps):
                        converged = True
                        break

                lambda_prev = current_eigvals
                Q, R = qr_decomposition_2x2(block)
                block = R @ Q

            if not converged:
                print(f"Блок 2×2 в позиции {i} не сошёлся за {max_iterations} итераций")

            eigenvalues.extend(lambda_prev)
            i += 2

    return np.array(eigenvalues)


def qr_decomposition_2x2(A):
    """QR-разложение 2x2 матрицы с использованием преобразований Хаусхолдера."""
    n = 2
    Q = np.eye(n)
    R = A.copy()

    a = R[:, 0]
    norm_a = euclidean_norm(a)
    v = a.copy()
    v[0] += np.sign(a[0]) * norm_a
    v = v / euclidean_norm(v)

    H = np.eye(n) - 2 * np.outer(v, v)
    R = H @ R
    Q = Q @ H

    return Q, R


def qr_algorithm(A, epsilon, max_iterations=1000):
    """Находит собственные значения и векторы матрицы A с использованием QR-алгоритма."""
    n = A.shape[0]
    Ak = A.copy()
    Q_total = np.eye(n)

    for _ in range(max_iterations):
        Q, R = qr_decomposition(Ak)
        Q_total = Q_total @ Q
        Ak = R @ Q

        # Проверка сходимости (сумма квадратов поддиагональных элементов)
        off_diag_sum = np.sqrt(np.sum(np.tril(Ak, -1)**2))
        if off_diag_sum < epsilon:
            break

    eigenvalues = extract_eigenvalues(Ak, eps=epsilon)
    eigenvectors = Q_total
    return eigenvalues, eigenvectors


A = np.array([
    [1, 7, -1],
    [-2, 2, -2],
    [9, -7, 3]
], dtype=float)

eigenvalues, eigenvectors = qr_algorithm(A, epsilon=0.01)

print("Собственные значения:")
for idx, val in enumerate(eigenvalues):
    if abs(val.imag) < 0.01:  # считаем значение вещественным, если мнимая часть близка к 0
        print(f"λ{idx + 1} = {val.real:.2f}")
    else:
        sign = '+' if val.imag >= 0 else '-'
        imag_part = abs(val.imag)
        print(f"λ{idx + 1} = {val.real:.2f} {sign} {imag_part:.2f}i")

print("\nСобственные векторы:")
for i, vec in enumerate(eigenvectors.T):
    formatted_vec = ', '.join(f"{val:.2f}" for val in vec)
    print(f"v{i + 1} = [{formatted_vec}]")