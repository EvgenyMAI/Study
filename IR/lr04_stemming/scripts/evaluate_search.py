import sys
import subprocess
import json

def search(query):
    """Execute search and return results"""
    result = subprocess.run([
        '../build/stemmer.exe', 'search',
        '../data/search_index.txt',
        query
    ], capture_output=True, text=True, encoding='utf-8')
    
    lines = result.stdout.strip().split('\n')
    
    if len(lines) < 1:
        return []
    
    try:
        num_results = int(lines[0])
        results = [int(lines[i+1]) for i in range(min(num_results, len(lines)-1))]
        return results
    except ValueError:
        return []

def calculate_precision_at_k(results, relevant, k=10):
    """Calculate P@K"""
    if len(results) == 0:
        return 0.0
    
    k = min(k, len(results))
    relevant_found = sum(1 for r in results[:k] if r in relevant)
    return relevant_found / k

def evaluate():
    """Evaluate search quality"""
    test_queries = [
        {
            'query': 'python программирование',
            'description': 'Технический запрос (должны быть Habr статьи)'
        },
        {
            'query': 'машинное обучение',
            'description': 'AI/ML тематика'
        },
        {
            'query': 'веб разработка',
            'description': 'Web development'
        },
        {
            'query': 'новости россия',
            'description': 'Новостной запрос (Lenta)'
        },
        {
            'query': 'технологии инновации',
            'description': 'Общая технологическая тематика'
        }
    ]
    
    print("=" * 60)
    print("ОЦЕНКА КАЧЕСТВА ПОИСКА СО СТЕММИНГОМ")
    print("=" * 60)
    print()
    
    all_results = []
    
    for i, test in enumerate(test_queries, 1):
        query = test['query']
        description = test['description']
        
        results = search(query)
        
        print(f"Запрос {i}: '{query}'")
        print(f"Описание: {description}")
        print(f"Найдено результатов: {len(results)}")
        
        if len(results) > 0:
            print(f"Топ-5 документов: {results[:5]}")
        else:
            print("Результатов не найдено!")
        
        print()
        
        all_results.append({
            'query': query,
            'num_results': len(results),
            'top5': results[:5] if len(results) > 0 else []
        })
    
    print("=" * 60)
    print("ВЫВОДЫ")
    print("=" * 60)
    print()
    print("Стемминг успешно работает:")
    
    avg_results = sum(r['num_results'] for r in all_results) / len(all_results)
    print(f"- Среднее кол-во результатов: {avg_results:.1f}")
    print(f"- Все запросы вернули результаты: {all(r['num_results'] > 0 for r in all_results)}")
    print()
    print("Качество ранжирования (TF):")
    print("- Используется простое TF-ранжирование")
    print("- Для улучшения нужно добавить IDF и нормализацию")
    print()
    
    # Save results
    results_data = {
        'queries': all_results,
        'avg_results': avg_results,
        'description': 'Stemming evaluation with simple TF ranking'
    }
    
    with open('../data/evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    print(f"Результаты сохранены: ../data/evaluation_results.json")

if __name__ == '__main__':
    evaluate()