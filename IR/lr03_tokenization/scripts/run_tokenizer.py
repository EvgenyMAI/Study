"""
Python-обвязка для токенизации документов из MongoDB
"""

import sys
import os
import subprocess
import json
import time
from pymongo import MongoClient

def connect_to_db():
    """Подключение к MongoDB"""
    try:
        client = MongoClient('localhost', 27017, serverSelectionTimeoutMS=5000)
        client.server_info()
        db = client['search_engine']
        collection = db['documents']
        return collection
    except Exception as e:
        print(f"Ошибка подключения к MongoDB: {e}")
        sys.exit(1)

def extract_text_from_html(html):
    """Простое извлечение текста из HTML (без BeautifulSoup для скорости)"""
    import re
    # Удаляем script и style
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
    # Удаляем теги
    text = re.sub(r'<[^>]+>', ' ', html)
    # Декодируем HTML entities
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    # Убираем множественные пробелы
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def prepare_corpus_file(collection, output_file, max_docs=None):
    """Подготовка файла с текстами для токенизации"""
    print("Извлечение текстов из MongoDB...")
    
    query = {}
    cursor = collection.find(query)
    
    if max_docs:
        cursor = cursor.limit(max_docs)
    
    doc_count = 0
    total_size = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in cursor:
            html = doc.get('html', '')
            text = extract_text_from_html(html)
            
            if text:
                f.write(text + '\n\n')
                doc_count += 1
                total_size += len(text)
                
                if doc_count % 1000 == 0:
                    print(f"Обработано документов: {doc_count}")
    
    print(f"\nПодготовлено документов: {doc_count}")
    print(f"Общий размер: {total_size / 1024:.2f} КБ")
    
    return doc_count, total_size

def run_tokenizer(input_file, output_file, tokenizer_path='../tokenizer/tokenizer.exe'):
    """Запуск C++ токенизатора"""
    print("\nЗапуск токенизатора...")
    
    if not os.path.exists(tokenizer_path):
        # Попробуем без .exe (Linux/Mac)
        tokenizer_path = tokenizer_path.replace('.exe', '')
    
    if not os.path.exists(tokenizer_path):
        print(f"Ошибка: токенизатор не найден по пути {tokenizer_path}")
        print("Сначала скомпилируйте проект: compile.bat")
        sys.exit(1)
    
    cmd = [tokenizer_path, '-s', input_file, output_file]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        print(result.stdout)
        if result.stderr:
            print("Ошибки:", result.stderr)
        
        return result.returncode == 0
    except Exception as e:
        print(f"Ошибка запуска токенизатора: {e}")
        return False

def analyze_tokens(tokens_file, output_json):
    """Анализ токенов и сохранение статистики"""
    print("\nАнализ токенов...")
    
    tokens = []
    token_counts = {}
    total_length = 0
    
    # Явно указываем UTF-8 и игнорируем ошибки
    with open(tokens_file, 'r', encoding='utf-8-sig', errors='ignore') as f:
        for line in f:
            token = line.strip()
            if token:
                tokens.append(token)
                total_length += len(token)
                token_counts[token] = token_counts.get(token, 0) + 1
    
    total_tokens = len(tokens)
    unique_tokens = len(token_counts)
    avg_length = total_length / total_tokens if total_tokens > 0 else 0
    
    # Топ-20 частых токенов
    top_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    
    # Статистика по длине токенов
    length_distribution = {}
    for token in tokens:
        length = len(token)
        length_distribution[length] = length_distribution.get(length, 0) + 1
    
    stats = {
        'total_tokens': total_tokens,
        'unique_tokens': unique_tokens,
        'average_length': round(avg_length, 2),
        'top_20_tokens': [{'token': t, 'count': c} for t, c in top_tokens],
        'length_distribution': dict(sorted(length_distribution.items()))
    }
    
    # Сохранение в JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"\nВсего токенов: {total_tokens:,}")
    print(f"Уникальных токенов: {unique_tokens:,}")
    print(f"Средняя длина токена: {avg_length:.2f} символов")
    print(f"\nТоп-10 частых токенов:")
    for token, count in top_tokens[:10]:
        print(f"  {token}: {count:,}")
    
    return stats

def main():
    """Главная функция"""
    print("="*60)
    print("ТОКЕНИЗАЦИЯ КОРПУСА ДОКУМЕНТОВ")
    print("="*60)
    
    # Параметры
    max_docs = 50000
    temp_text_file = '../output/corpus_text.txt'
    tokens_file = '../output/tokens.txt'
    stats_file = '../output/statistics.json'
    
    # Создание папки output
    os.makedirs('../output', exist_ok=True)
    
    # Шаг 1: Подключение к БД
    collection = connect_to_db()
    
    # Шаг 2: Извлечение текстов
    start_time = time.time()
    doc_count, text_size = prepare_corpus_file(collection, temp_text_file, max_docs)
    
    # Шаг 3: Токенизация
    success = run_tokenizer(temp_text_file, tokens_file)
    
    if not success:
        print("Ошибка токенизации")
        sys.exit(1)
    
    # Шаг 4: Анализ
    stats = analyze_tokens(tokens_file, stats_file)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n{'='*60}")
    print(f"ИТОГО:")
    print(f"Документов обработано: {doc_count}")
    print(f"Общее время: {total_time:.2f} сек")
    print(f"Скорость: {doc_count / total_time:.1f} док/сек")
    print(f"Статистика сохранена: {stats_file}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()