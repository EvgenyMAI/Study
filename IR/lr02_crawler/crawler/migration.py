import os
import sys
import json
import yaml
import hashlib
import time
from datetime import datetime
from urllib.parse import urlparse, urlunparse
from pymongo import MongoClient, UpdateOne
from tqdm import tqdm  # pip install tqdm

# Настройки путей
CORPUS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../lr01_corpus/corpus'))
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def normalize_url(url):
    """Точная копия логики из crawler.py"""
    parsed = urlparse(url)
    normalized = urlunparse((
        parsed.scheme,
        parsed.netloc.lower(),
        parsed.path,
        parsed.params,
        parsed.query,
        '' 
    ))
    if normalized.endswith('/') and normalized.count('/') > 3:
        normalized = normalized[:-1]
    return normalized

def calculate_hash(html):
    return hashlib.md5(html.encode('utf-8')).hexdigest()

def migrate():
    print(f"Загрузка конфигурации из {CONFIG_PATH}...")
    config = load_config(CONFIG_PATH)
    
    # Подключение к БД
    client = MongoClient(config['db']['host'], config['db']['port'])
    db = client[config['db']['database']]
    collection = db[config['db']['collection']]
    
    print(f"Подключено к БД: {config['db']['database']}")
    print(f"Ищем корпус в: {CORPUS_DIR}")

    if not os.path.exists(CORPUS_DIR):
        print(f"ОШИБКА: Папка корпуса не найдена: {CORPUS_DIR}")
        sys.exit(1)

    sources = ['habr', 'lenta']
    total_inserted = 0
    total_errors = 0

    for source in sources:
        source_path = os.path.join(CORPUS_DIR, source)
        metadata_path = os.path.join(source_path, 'metadata.json')
        raw_dir = os.path.join(source_path, 'raw')

        if not os.path.exists(metadata_path):
            print(f"Пропуск источника {source}: нет metadata.json")
            continue

        print(f"\nОбработка источника: {source}...")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        operations = []
        batch_size = 500
        
        for item in tqdm(metadata, desc=f"Миграция {source}"):
            try:
                # 1. Читаем HTML файл
                file_id = item.get('id', str(item.get('doc_id', ''))) # Обработка разных форматов ID
                html_path = os.path.join(raw_dir, f"{file_id}.html")
                
                if not os.path.exists(html_path):
                    # Fallback: иногда ID в метаданных отличается от имени файла, проверяем
                    continue

                with open(html_path, 'r', encoding='utf-8') as hf:
                    html_content = hf.read()

                # 2. Подготовка полей
                url = normalize_url(item['url'])
                content_hash = calculate_hash(html_content)
                
                # Парсинг даты из ISO в Unix timestamp
                try:
                    dt = datetime.fromisoformat(item['date'].replace('Z', '+00:00'))
                    crawled_at = int(dt.timestamp())
                except:
                    crawled_at = int(time.time())

                doc = {
                    'url': url,
                    'html': html_content,
                    'source': source,
                    'crawled_at': crawled_at,
                    'content_hash': content_hash,
                    'size': len(html_content)
                }

                # 3. Добавляем операцию upsert (обновить если есть, вставить если нет)
                operations.append(
                    UpdateOne({'url': url}, {'$set': doc}, upsert=True)
                )

                # 4. Выполняем пачками
                if len(operations) >= batch_size:
                    collection.bulk_write(operations)
                    total_inserted += len(operations)
                    operations = []

            except Exception as e:
                # print(f"Ошибка с документом {item.get('url')}: {e}")
                total_errors += 1

        # Дозаписываем остатки
        if operations:
            collection.bulk_write(operations)
            total_inserted += len(operations)

    print("\n" + "="*40)
    print(f"Миграция завершена.")
    print(f"Успешно обработано: {total_inserted}")
    print(f"Ошибок чтения: {total_errors}")
    print(f"Всего документов в БД: {collection.count_documents({})}")
    print("="*40)

if __name__ == '__main__':
    migrate()