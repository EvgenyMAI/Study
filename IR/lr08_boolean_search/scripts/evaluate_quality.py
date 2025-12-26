import subprocess
import os
import json

def search(query):
    """Выполнить поисковый запрос"""
    exe_path = os.path.join(os.path.dirname(__file__), '..', 'build', 'boolean_search.exe')
    index_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'boolean_index.txt')
    
    result = subprocess.run(
        [exe_path, index_path, query],
        capture_output=True,
        text=True,
        encoding='utf-8'
    )
    
    lines = result.stdout.strip().split('\n')
    if len(lines) < 1:
        return []
    
    try:
        num_results = int(lines[0])
        doc_ids = [int(lines[i+1]) for i in range(min(num_results, len(lines)-1))]
        return doc_ids
    except (ValueError, IndexError):
        return []

def evaluate():
    """Оценка качества булева поиска"""
    
    test_queries = [
        {
            'query': 'python',
            'description': 'Простой запрос',
            'expected_min': 100
        },
        {
            'query': 'python AND программирование',
            'description': 'AND запрос',
            'expected_min': 10
        },
        {
            'query': 'python OR java',
            'description': 'OR запрос',
            'expected_min': 100
        },
        {
            'query': 'машинное AND обучение',
            'description': 'Русский AND',
            'expected_min': 50
        },
        {
            'query': 'python AND NOT javascript',
            'description': 'NOT запрос',
            'expected_min': 50
        },
        {
            'query': 'веб разработка',
            'description': 'Неявный AND',
            'expected_min': 10
        }
    ]
    
    print("=" * 70)
    print("ОЦЕНКА КАЧЕСТВА БУЛЕВА ПОИСКА")
    print("=" * 70)
    print()
    
    results = []
    
    for i, test_case in enumerate(test_queries, 1):
        query = test_case['query']
        description = test_case['description']
        expected_min = test_case['expected_min']
        
        print(f"Запрос {i}: '{query}'")
        print(f"Описание: {description}")
        
        doc_ids = search(query)
        
        print(f"Найдено: {len(doc_ids)} документов")
        print(f"Топ-5: {doc_ids[:5]}")
        
        status = "Ok" if len(doc_ids) >= expected_min else "No"
        print(f"Ожидалось минимум: {expected_min} {status}")
        print()
        
        results.append({
            'query': query,
            'description': description,
            'num_results': len(doc_ids),
            'top5': doc_ids[:5],
            'expected_min': expected_min,
            'passed': len(doc_ids) >= expected_min
        })
    
    # Сохранение результатов
    output_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'evaluation_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("=" * 70)
    print("ИТОГИ")
    print("=" * 70)
    passed = sum(1 for r in results if r['passed'])
    print(f"Пройдено тестов: {passed}/{len(results)}")
    print(f"Результаты сохранены: {output_file}")
    print()

if __name__ == '__main__':
    evaluate()