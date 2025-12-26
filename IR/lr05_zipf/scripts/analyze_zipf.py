"""
Анализ отклонений от закона Ципфа
"""

import json
import numpy as np
from pathlib import Path

def analyze_deviations(data):
    """Анализ отклонений"""
    frequencies_list = data['frequencies']
    
    ranks = np.array([item['rank'] for item in frequencies_list])
    frequencies = np.array([item['frequency'] for item in frequencies_list])
    terms = [item['term'] for item in frequencies_list]
    
    # Закон Ципфа
    C = frequencies[0]
    zipf_predicted = C / ranks
    
    # Относительные отклонения
    rel_deviations = (frequencies - zipf_predicted) / zipf_predicted * 100
    
    # Анализ зон
    print("\n" + "=" * 60)
    print("АНАЛИЗ ОТКЛОНЕНИЙ ОТ ЗАКОНА ЦИПФА")
    print("=" * 60)
    
    print("\n1. Высокочастотная зона (топ-100):")
    high_freq_dev = np.mean(np.abs(rel_deviations[:100]))
    print(f"Среднее отклонение: {high_freq_dev:.1f}%")
    print("Причины: стоп-слова ('в', 'и', 'не') превышают ожидания")
    
    print("\n2. Среднечастотная зона (100-10000):")
    mid_freq_dev = np.mean(np.abs(rel_deviations[100:10000]))
    print(f"Среднее отклонение: {mid_freq_dev:.1f}%")
    print("Причины: близко к теории, основная масса терминов")
    
    print("\n3. Низкочастотная зона (>10000):")
    if len(rel_deviations) > 10000:
        low_freq_dev = np.mean(np.abs(rel_deviations[10000:]))
        print(f"Среднее отклонение: {low_freq_dev:.1f}%")
        print("Причины: редкие термины, опечатки, имена собственные")
    
    # Топ отклонений
    print("\n4. Топ-10 положительных отклонений (частота выше ожидаемой):")
    top_positive = np.argsort(rel_deviations)[-10:][::-1]
    for i in top_positive:
        if rel_deviations[i] > 0:
            print(f"   {terms[i]:20s} : +{rel_deviations[i]:.1f}% (ранг {ranks[i]})")
    
    # Сохранение отчёта
    report_file = Path('../output/zipf_analysis.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("АНАЛИЗ ОТКЛОНЕНИЙ ОТ ЗАКОНА ЦИПФА\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Высокочастотная зона: {high_freq_dev:.1f}%\n")
        f.write(f"Среднечастотная зона: {mid_freq_dev:.1f}%\n")
        if len(rel_deviations) > 10000:
            f.write(f"Низкочастотная зона: {low_freq_dev:.1f}%\n")
    
    print(f"\nОтчёт сохранён: {report_file}")

def main():
    current_dir = Path(__file__).parent
    freq_file = current_dir.parent / 'output' / 'term_frequencies.json'
    
    with open(freq_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    analyze_deviations(data)

if __name__ == '__main__':
    main()