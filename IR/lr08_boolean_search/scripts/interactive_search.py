import subprocess
import os
import sys

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
        print(f"Error parsing results")
        return []

def main():
    print("=" * 70)
    print("ИНТЕРАКТИВНЫЙ БУЛЕВ ПОИСК")
    print("=" * 70)
    print()
    print("Операторы: AND, OR, NOT")
    print("Примеры:")
    print("python AND программирование")
    print("машинное обучение")
    print("python OR java")
    print("python AND NOT javascript")
    print()
    print("Введите 'exit' для выхода")
    print("=" * 70)
    print()
    
    while True:
        try:
            query = input("Запрос> ").strip()
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("Выход...")
                break
            
            if not query:
                continue
            
            results = search(query)
            
            print(f"\nНайдено: {len(results)} документов")
            if len(results) > 0:
                print(f"Топ-10: {results[:10]}")
            print()
            
        except KeyboardInterrupt:
            print("\n\nВыход...")
            break
        except EOFError:
            break

if __name__ == '__main__':
    main()