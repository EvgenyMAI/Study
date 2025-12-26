"""
Подсчёт частот терминов для закона Ципфа
"""

import json
import sys
from collections import Counter
from pathlib import Path
from tqdm import tqdm

sys.setrecursionlimit(2000)

def get_tokens_generator(tokens_file):
    """
    Генератор для потокового чтения токенов.
    Использует errors='replace', чтобы не падать на битых байтах,
    а сохранять корректную кириллицу.
    """
    with open(tokens_file, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            token = line.strip()
            if token:
                yield token

def filter_garbage(token):
    """Проверка на мусор и стоп-слова"""
    garbage = {
        'нравится', 'добавить', 'закладки', 'комментарий', 'ответить', 
        'рейтинг', 'скрыт', 'избранное', 'поделиться', 'подписаться',
        'читать', 'далее', 'продолжение', 'источник', ''
    }
    # Также фильтруем токены, ставшие "вопросиками" из-за ошибок кодировки
    return token not in garbage

def calculate_frequencies_stream(tokens_file):
    """Потоковый подсчёт частот"""
    print("\nПодсчёт частот (потоковая обработка, UTF-8)...")
    
    counter = Counter()
    
    # Оценка количества строк для прогресс-бара (грубо)
    # Средний размер токена на русском ~12 байт (с переносом)
    total_lines_approx = tokens_file.stat().st_size // 12
    
    token_gen = get_tokens_generator(tokens_file)
    
    for token in tqdm(token_gen, total=total_lines_approx, unit="tok", mininterval=1.0):
        if filter_garbage(token):
            counter[token] += 1

    # Сортировка по убыванию частоты
    print("\nСортировка результатов...")
    sorted_terms = counter.most_common()
    
    print(f"Уникальных терминов: {len(sorted_terms):,}")
    print(f"Всего токенов (учтенных): {sum(counter.values()):,}")
    
    print(f"\nТоп-20 терминов:")
    for i, (term, freq) in enumerate(sorted_terms[:20], 1):
        print(f"  {i:2d}. {term:20s} : {freq:,}")
    
    return sorted_terms

def save_frequencies(sorted_terms, output_file):
    """Сохранение частот в JSON"""
    print(f"\nСохранение в {output_file}...")
    
    data = {
        'total_tokens': sum(freq for _, freq in sorted_terms),
        'unique_terms': len(sorted_terms),
        'frequencies': [
            {
                'rank': i,
                'term': term,
                'frequency': freq
            }
            for i, (term, freq) in enumerate(sorted_terms, 1)
        ]
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Сохранено {len(sorted_terms):,} терминов")

def main():
    # Автоопределение путей
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    
    # Ищем tokens.txt в разных местах
    possible_paths = [
        project_root / 'lr03_tokenization' / 'output' / 'tokens.txt',
        project_root / 'output' / 'tokens.txt',
        Path('../../lr03_tokenization/output/tokens.txt'),
        Path('../../output/tokens.txt')
    ]
    
    tokens_file = None
    for p in possible_paths:
        if p.exists():
            tokens_file = p
            break
            
    if not tokens_file:
        raise FileNotFoundError("Файл tokens.txt не найден! Проверьте пути.")
    
    output_file = current_dir.parent / 'output' / 'term_frequencies.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("ПОДСЧЁТ ЧАСТОТ ТЕРМИНОВ (Fixed UTF-8)")
    print("=" * 60)
    print(f"Токены: {tokens_file.absolute()}")
    print(f"Размер файла: {tokens_file.stat().st_size / (1024*1024):.2f} MB")
    print(f"Вывод:  {output_file.absolute()}")
    print()
    
    sorted_terms = calculate_frequencies_stream(tokens_file)
    save_frequencies(sorted_terms, output_file)
    
    print("\n" + "=" * 60)
    print("Готово!")
    print("=" * 60)

if __name__ == '__main__':
    main()