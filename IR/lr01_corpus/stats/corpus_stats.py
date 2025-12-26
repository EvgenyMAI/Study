import os
import json
from pathlib import Path

def calculate_statistics(corpus_dir):
    """Вычисляет статистику по корпусу"""
    
    sources = ['habr', 'lenta']
    total_stats = {
        'total_documents': 0,
        'total_raw_size': 0,
        'total_text_size': 0,
        'sources': {}
    }
    
    for source in sources:
        source_dir = os.path.join(corpus_dir, source)
        
        if not os.path.exists(source_dir):
            continue
        
        # Загружаем метаданные
        metadata_file = os.path.join(source_dir, 'metadata.json')
        if not os.path.exists(metadata_file):
            continue
            
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Подсчитываем статистику
        num_docs = len(metadata)
        total_raw = sum(item['raw_size'] for item in metadata)
        total_text = sum(item['text_length'] for item in metadata)
        
        avg_raw = total_raw / num_docs if num_docs > 0 else 0
        avg_text = total_text / num_docs if num_docs > 0 else 0
        
        source_stats = {
            'documents': num_docs,
            'total_raw_size_bytes': total_raw,
            'total_raw_size_mb': round(total_raw / (1024 * 1024), 2),
            'total_text_size_bytes': total_text,
            'total_text_size_mb': round(total_text / (1024 * 1024), 2),
            'avg_raw_size_bytes': round(avg_raw, 2),
            'avg_raw_size_kb': round(avg_raw / 1024, 2),
            'avg_text_size_bytes': round(avg_text, 2),
            'avg_text_size_kb': round(avg_text / 1024, 2)
        }
        
        total_stats['sources'][source] = source_stats
        total_stats['total_documents'] += num_docs
        total_stats['total_raw_size'] += total_raw
        total_stats['total_text_size'] += total_text
    
    # Общая статистика
    if total_stats['total_documents'] > 0:
        total_stats['total_raw_size_mb'] = round(total_stats['total_raw_size'] / (1024 * 1024), 2)
        total_stats['total_text_size_mb'] = round(total_stats['total_text_size'] / (1024 * 1024), 2)
        total_stats['avg_raw_size_kb'] = round(total_stats['total_raw_size'] / total_stats['total_documents'] / 1024, 2)
        total_stats['avg_text_size_kb'] = round(total_stats['total_text_size'] / total_stats['total_documents'] / 1024, 2)
    
    return total_stats

def print_statistics(stats):
    """Выводит статистику в читаемом виде"""
    print("\n" + "="*60)
    print("СТАТИСТИКА КОРПУСА ДОКУМЕНТОВ")
    print("="*60)
    
    print(f"\nОБЩАЯ СТАТИСТИКА:")
    print(f"Всего документов: {stats['total_documents']}")
    print(f"Общий размер сырых данных: {stats.get('total_raw_size_mb', 0)} МБ")
    print(f"Общий размер текста: {stats.get('total_text_size_mb', 0)} МБ")
    print(f"Средний размер сырого документа: {stats.get('avg_raw_size_kb', 0)} КБ")
    print(f"Средний размер текста документа: {stats.get('avg_text_size_kb', 0)} КБ")
    
    for source, source_stats in stats['sources'].items():
        print(f"\n{source.upper()}:")
        print(f"Документов: {source_stats['documents']}")
        print(f"Размер сырых данных: {source_stats['total_raw_size_mb']} МБ")
        print(f"Размер текста: {source_stats['total_text_size_mb']} МБ")
        print(f"Средний размер сырого документа: {source_stats['avg_raw_size_kb']} КБ")
        print(f"Средний размер текста: {source_stats['avg_text_size_kb']} КБ")
    
    print("\n" + "="*60)

def save_statistics(stats, output_file):
    """Сохраняет статистику в JSON"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\nСтатистика сохранена в {output_file}")

if __name__ == '__main__':
    corpus_dir = '../corpus'
    stats = calculate_statistics(corpus_dir)
    print_statistics(stats)
    
    stats_file = '../stats/corpus_statistics.json'
    os.makedirs(os.path.dirname(stats_file), exist_ok=True)
    save_statistics(stats, stats_file)