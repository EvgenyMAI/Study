"""
Детальный анализ токенов с примерами проблем
"""

import json
import sys
from collections import Counter

def load_tokens(filename):
    """Загрузка токенов из файла"""
    tokens = []
    with open(filename, 'r', encoding='utf-8-sig', errors='ignore') as f:
        for line in f:
            token = line.strip()
            if token:
                tokens.append(token)
    return tokens

def find_problematic_tokens(tokens):
    """Поиск проблемных токенов"""
    problems = {
        'only_digits': [],
        'too_short': [],
        'too_long': [],
        'mixed_scripts': [],
        'with_special_chars': []
    }
    
    for token in set(tokens[:10000]):  # Анализируем первые 10k уникальных
        # Только цифры
        if token.isdigit():
            problems['only_digits'].append(token)
        
        # Слишком короткие (1-2 символа)
        if len(token) <= 2:
            problems['too_short'].append(token)
        
        # Слишком длинные (>30 символов, возможно ошибка)
        if len(token) > 30:
            problems['too_long'].append(token)
        
        # Смешение кириллицы и латиницы
        has_cyrillic = any('\u0400' <= c <= '\u04FF' for c in token)
        has_latin = any('a' <= c.lower() <= 'z' for c in token)
        if has_cyrillic and has_latin:
            problems['mixed_scripts'].append(token)
        
        # Специальные символы внутри слова (кроме дефиса)
        if any(c in token for c in ['_', '+', '=', '@', '#']):
            problems['with_special_chars'].append(token)
    
    return problems

def print_problems(problems):
    """Вывод проблемных токенов"""
    print("\n" + "="*60)
    print("ПРОБЛЕМНЫЕ ТОКЕНЫ")
    print("="*60)
    
    print("\n1. Только цифры (первые 10):")
    for token in problems['only_digits'][:10]:
        print(f"   {token}")
    print(f"   Всего: {len(problems['only_digits'])}")
    
    print("\n2. Очень короткие (≤2 символов, первые 10):")
    for token in problems['too_short'][:10]:
        print(f"   '{token}'")
    print(f"   Всего: {len(problems['too_short'])}")
    
    print("\n3. Очень длинные (>30 символов, первые 10):")
    for token in problems['too_long'][:10]:
        print(f"   {token[:50]}...")
    print(f"   Всего: {len(problems['too_long'])}")
    
    print("\n4. Смешение алфавитов (первые 10):")
    for token in problems['mixed_scripts'][:10]:
        print(f"   {token}")
    print(f"   Всего: {len(problems['mixed_scripts'])}")
    
    print("\n5. Специальные символы (первые 10):")
    for token in problems['with_special_chars'][:10]:
        print(f"   {token}")
    print(f"   Всего: {len(problems['with_special_chars'])}")

def main():
    tokens_file = '../output/tokens.txt'
    
    print("Загрузка токенов...")
    tokens = load_tokens(tokens_file)
    
    print(f"Загружено токенов: {len(tokens):,}")
    
    # Поиск проблем
    problems = find_problematic_tokens(tokens)
    print_problems(problems)
    
    print("\n" + "="*60)

if __name__ == '__main__':
    main()
