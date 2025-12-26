"""
Построение графика закона Ципфа
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Настройка для русского языка
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_frequencies(freq_file):
    """Загрузка частот из JSON"""
    with open(freq_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def zipf_law(rank, C=None, total_freq=None):
    """
    Закон Ципфа: freq(r) = C / r
    где C - константа, r - ранг термина
    """
    if C is None:
        # Подбираем C из первого термина
        C = total_freq
    return C / rank

def mandelbrot_law(rank, C, B, alpha):
    """
    Закон Мандельброта: freq(r) = C / (r + B)^alpha
    Более гибкая модель, учитывает отклонения
    """
    return C / np.power(rank + B, alpha)

def fit_zipf_constant(ranks, frequencies):
    """Подбор константы C для закона Ципфа"""
    # C = freq(1) * 1 (из первого термина)
    C = frequencies[0]
    return C

def fit_mandelbrot_constants(ranks, frequencies):
    """
    Упрощённый подбор констант для Мандельброта
    В реальности нужна оптимизация (scipy.optimize)
    """
    # Начальные приближения
    C = frequencies[0]
    B = 2.7  # типичное значение
    alpha = 1.0  # для Ципфа alpha=1
    
    # Можно улучшить через minimize, но для ЛР достаточно фиксированных
    return C, B, alpha

def plot_zipf(data, output_file):
    """Построение графика"""
    frequencies_list = data['frequencies']
    
    ranks = np.array([item['rank'] for item in frequencies_list])
    frequencies = np.array([item['frequency'] for item in frequencies_list])
    
    # Подбор констант
    C_zipf = fit_zipf_constant(ranks, frequencies)
    C_mand, B_mand, alpha_mand = fit_mandelbrot_constants(ranks, frequencies)
    
    # Теоретические кривые
    zipf_predicted = zipf_law(ranks, C=C_zipf)
    mandelbrot_predicted = mandelbrot_law(ranks, C_mand, B=B_mand, alpha=alpha_mand)
    
    # График
    plt.figure(figsize=(12, 8))
    
    # Логарифмическая шкала
    plt.loglog(ranks, frequencies, 'o', markersize=2, alpha=0.6, label='Реальные данные')
    plt.loglog(ranks, zipf_predicted, 'r-', linewidth=2, label=f'Закон Ципфа: C={C_zipf:.0f}')
    plt.loglog(ranks, mandelbrot_predicted, 'g--', linewidth=2, 
               label=f'Закон Мандельброта: C={C_mand:.0f}, B={B_mand:.1f}, α={alpha_mand:.2f}')
    
    plt.xlabel('Ранг термина (log)', fontsize=12)
    plt.ylabel('Частота (log)', fontsize=12)
    plt.title('Распределение терминов: Закон Ципфа vs реальные данные', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"График сохранён: {output_file}")
    
    # Статистика отклонений
    mse_zipf = np.mean((frequencies - zipf_predicted) ** 2)
    mse_mand = np.mean((frequencies - mandelbrot_predicted) ** 2)
    
    print(f"\nСреднеквадратичная ошибка:")
    print(f"  Закон Ципфа: {mse_zipf:.2e}")
    print(f"  Закон Мандельброта: {mse_mand:.2e}")
    
    return C_zipf, (C_mand, B_mand, alpha_mand)

def main():
    current_dir = Path(__file__).parent
    freq_file = current_dir.parent / 'output' / 'term_frequencies.json'
    output_plot = current_dir.parent / 'output' / 'zipf_plot.png'
    
    print("=" * 60)
    print("ПОСТРОЕНИЕ ГРАФИКА ЗАКОНА ЦИПФА")
    print("=" * 60)
    
    data = load_frequencies(freq_file)
    
    print(f"\nВсего токенов: {data['total_tokens']:,}")
    print(f"Уникальных терминов: {data['unique_terms']:,}")
    
    C_zipf, (C_mand, B_mand, alpha_mand) = plot_zipf(data, output_plot)
    
    print("\n" + "=" * 60)
    print("КОНСТАНТЫ:")
    print(f"Закон Ципфа: C = {C_zipf:.0f}")
    print(f"Закон Мандельброта: C = {C_mand:.0f}, B = {B_mand:.1f}, α = {alpha_mand:.2f}")
    print("=" * 60)

if __name__ == '__main__':
    main()