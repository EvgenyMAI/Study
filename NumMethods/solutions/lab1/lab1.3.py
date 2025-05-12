import numpy as np

def transform_system(A, b):
    '''
    Преобразует систему Ax = b в вид x = αx + β.
    '''
    n = len(b)
    alpha = -A / A.diagonal()[:, None]  # Делим каждую строку на A_ii и меняем знак
    np.fill_diagonal(alpha, 0)  # Зануляем главную диагональ (A_ii -> 0)
    beta = b / A.diagonal()
    return alpha, beta

def calculate_matrix_norm(alpha):
    '''
    Вычисляет норму матрицы
    '''
    norms = {
        '1': np.max(np.sum(np.abs(alpha), axis=0)),  # Столбцовая норма
        'C': np.max(np.sum(np.abs(alpha), axis=1)),  # Строчная норма
        '2': np.sqrt(np.sum(alpha**2))
    }
    norm_type, min_norm = min(norms.items(), key=lambda x: x[1])
    print(f"Выбрана норма {norm_type} для матрицы альфа")
    return min_norm, norm_type

def calculate_vector_norm(vec, norm_type):
    '''
    Вычисляет норму вектора в соответствии с нормой матрицы.
    '''
    return {
        '1': np.sum(np.abs(vec)),
        '2': np.sqrt(np.sum(vec**2)),
        'C': np.max(np.abs(vec))
    }[norm_type]

def check_convergence(x, x_prev, epsilon, norm_alpha, norm_type, use_matrix_norm):
    '''
    Проверяет условие сходимости:
    - Если use_matrix_norm=True, учитывается норма матрицы.
    - Если False, используется только норма разности векторов.
    '''
    error_norm = calculate_vector_norm(x - x_prev, norm_type)
    return (norm_alpha * error_norm / (1 - norm_alpha)) <= epsilon if use_matrix_norm else error_norm <= epsilon

def iterative_solver(alpha, beta, epsilon, norm_alpha, norm_type, update_rule, use_matrix_norm):
    '''
    Универсальный итерационный метод (простые итерации / Зейдель).
    update_rule: функция, определяющая обновление x (обычные итерации / Зейдель).
    '''
    n = len(beta)
    x = np.zeros(n)
    iterations = 0

    while True:
        x_prev = x.copy()
        x = update_rule(alpha, beta, x, x_prev)
        iterations += 1
        
        if check_convergence(x, x_prev, epsilon, norm_alpha, norm_type, use_matrix_norm):
            break

    return x, iterations

def simple_iteration_update(alpha, beta, x, x_prev):
    '''Правило обновления для метода простых итераций'''
    return beta + np.dot(alpha, x_prev)

def seidel_update(alpha, beta, x, x_prev):
    '''Правило обновления для метода Зейделя'''
    n = len(beta)
    for i in range(n):
        sum1 = np.dot(alpha[i, :i], x[:i])
        sum2 = np.dot(alpha[i, i+1:], x_prev[i+1:])
        x[i] = beta[i] + sum1 + sum2
    return x

if __name__ == "__main__":
    A = np.array([
        [-14, 6, 1, -5],
        [-6, 27, 7, -6],
        [7, -5, -23, -8],
        [3, -8, -7, 26]
    ])
    
    b = np.array([95, -41, 69, 27])
    epsilon = 0.01

    alpha, beta = transform_system(A, b)

    norm_alpha, norm_type = calculate_matrix_norm(alpha)

    np.set_printoptions(suppress=True)

    use_matrix_norm = norm_alpha < 1
    solution_simple, iter_simple = iterative_solver(alpha, beta, epsilon, norm_alpha, norm_type, simple_iteration_update, use_matrix_norm)
    print(f"Решение (простые итерации): {solution_simple}, итераций: {iter_simple}")

    solution_seidel, iter_seidel = iterative_solver(alpha, beta, epsilon, norm_alpha, norm_type, seidel_update, use_matrix_norm=False)
    print(f"Решение (Зейдель): {solution_seidel}, итераций: {iter_seidel}")