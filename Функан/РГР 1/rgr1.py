import numpy as np
import matplotlib.pyplot as plt
import math
from typing import List

class SquashMapSolver:
    def __init__(self, density=1000, alpha=1/8):
        """
        Инициализирует класс.
        """
        self.density = density
        self.alpha = alpha
        self.T = np.linspace(0, 1, density)
    
    def apply_func(self, func: np.array, t: float) -> float:
        """
        Применяет функцию к заданному значению t, интерполируя по ближайшему индексу.
        """
        idx = np.searchsorted(self.T, t, side='right')
        return func[min(idx, len(func) - 1)]
    
    def F1(self, X: np.array, t: float) -> float:
        """
        Вычисляет F1(X, t) = 1/8 * x(3t) + 5.
        """
        return 1/8 * self.apply_func(X, 3*t) + 5
    
    def F2(self, X: np.array, t: float) -> float:
        """
        Вычисляет F2(X, t) = 1/8 * x(3t - 2) + 5.
        """
        return 1/8 * self.apply_func(X, 3*t - 2) + 5
    
    def between(self, x1: float, x2: float, p: float) -> float:
        """
        Линейно интерполирует значение между x1 и x2.
        """
        return x1 + (x2 - x1) * p
    
    def T_map(self, X: np.array) -> np.array:
        """
        Строит отображение T(x) для массива X.
        """
        T_X = np.zeros_like(X)
        for idx, t in enumerate(self.T):
            if t < 1/3:
                T_X[idx] = self.F1(X, t)
            elif t < 2/3:
                if t < 4/9:
                    T_X[idx] = self.between(self.F1(X, 1/3), 1, (t - 1/3) / (1/9))
                elif t < 5/9:
                    T_X[idx] = self.between(1, -1, (t - 4/9) / (1/9))
                else:
                    T_X[idx] = self.between(-1, self.F2(X, 2/3), (t - 5/9) / (1/9))
            else:
                T_X[idx] = self.F2(X, t)
        return T_X
    
    def distance(self, X: np.array, Y: np.array) -> float:
        """
        Вычисляет максимальное расстояние между массивами X и Y.
        """
        return np.max(np.abs(X - Y))
    
    def iterations_count(self, d0: float, eps: float) -> int:
        """
        Вычисляет необходимое количество итераций метода сжимающего отображения.
        """
        return int(np.ceil(math.log((1 - self.alpha) * eps / d0, self.alpha)))
    
    def squash_map_method(self, X0: np.array, eps: float) -> tuple[np.array, int]:
        """
        Выполняет метод сжимающего отображения для X0 с точностью eps.
        Возвращает найденную фиксированную точку и количество итераций.
        """
        X1 = self.T_map(X0)
        d0 = self.distance(X0, X1)
        iter_count = self.iterations_count(d0, eps)
        X_I = np.copy(X1)
        for _ in range(iter_count - 1):
            X_I = self.T_map(X_I)
        return X_I, iter_count
    
    def plot_graph(self, Xs: List[np.array], labels: List[str], colors: List[str], eps: float, title: str):
        """
        Строит график для переданных массивов Xs.
        """
        plt.figure(figsize=(8, 5))
        for X, label, color in zip(Xs, labels, colors):
            plt.plot(self.T, X, label=label, color=color, linewidth=2)
        plt.xlabel("t")
        plt.ylabel("x")
        plt.title(title)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.text(0.632, 0.355, f"ε = {eps}", transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.6))
        plt.show()

def main():
    solver = SquashMapSolver()

    # Задание точности
    eps = 1e-1

    # Задание X0 = 0
    X0 = np.zeros_like(solver.T)  # Или np.copy(solver.T) для X0 = t

    X1 = solver.T_map(X0)
    X_final, iter_count = solver.squash_map_method(X0, eps)
    
    solver.plot_graph([X0, X1, X_final],
                      labels=["Начальное X0", "Первая итерация", f"Последняя итерация ({iter_count})"],
                      colors=["black", "red", "blue"],
                      eps=eps,
                      title="Сравнение решений")

if __name__ == '__main__':
    main()