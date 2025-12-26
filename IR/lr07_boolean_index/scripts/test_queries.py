import sys
import os
import subprocess
import time

def search(query):
    """Execute boolean search"""
    exe_path = os.path.join(os.path.dirname(__file__), '..', 'build', 'boolean_index.exe')
    index_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'boolean_index.txt')

    # Проверяем что файлы существуют
    if not os.path.exists(exe_path):
        print(f"ERROR: Executable not found: {exe_path}")
        return [], 0, 0

    if not os.path.exists(index_path):
        print(f"ERROR: Index file not found: {index_path}")
        return [], 0, 0

    start_time = time.time()

    try:
        result = subprocess.run([
            exe_path, 'search',
            index_path,
            query
        ], capture_output=True, text=True, encoding='utf-8', timeout=300)  # 5 min timeout
    except subprocess.TimeoutExpired:
        print("ERROR: Search timed out after 5 minutes")
        return [], 0, 300
    except Exception as e:
        print(f"ERROR: Subprocess failed: {e}")
        return [], 0, 0

    elapsed_time = time.time() - start_time

    # Выводим stderr для диагностики
    if result.stderr:
        stderr_lines = result.stderr.strip().split('\n')
        # Показываем только важные строки (не все "Loaded X/Y terms...")
        for line in stderr_lines:
            if 'Error' in line or 'Loading index' in line or 'Index loaded' in line or 'Executing query' in line or 'Search completed' in line or 'Found' in line:
                print(line)

    # Проверяем return code
    if result.returncode != 0:
        print(f"ERROR: Program returned error code {result.returncode}")
        print(f"stdout: {result.stdout[:500]}")
        return [], 0, elapsed_time

    # Парсим результаты
    lines = result.stdout.strip().split('\n')

    # Отладочный вывод
    if len(lines) < 1 or not lines[0].strip():
        print(f"ERROR: Empty output from program")
        print(f"stdout length: {len(result.stdout)}")
        print(f"stdout preview: {result.stdout[:200]}")
        return [], 0, elapsed_time

    try:
        total_count = int(lines[0].strip())
        results = []

        # Парсим результаты, пропуская пустые строки
        for i in range(1, len(lines)):
            line = lines[i].strip()
            if line:  # Пропускаем пустые строки
                try:
                    doc_id = int(line)
                    results.append(doc_id)
                    if len(results) >= 10:  # Берем только первые 10
                        break
                except ValueError:
                    # Пропускаем строки которые не являются числами
                    continue

        return results, total_count, elapsed_time
    except (ValueError, IndexError) as e:
        print(f"ERROR: Failed to parse results for query: {query}")
        print(f"Error: {e}")
        print(f"First line: '{lines[0] if lines else 'EMPTY'}'")
        print(f"Total lines: {len(lines)}")
        return [], 0, elapsed_time

def test_queries():
    """Test various boolean queries"""

    test_cases = [
        {
            'query': 'python',
            'description': 'Simple single-term query'
        },
        {
            'query': 'python AND программирование',
            'description': 'AND query (both terms must appear)'
        },
        {
            'query': 'python OR java',
            'description': 'OR query (either term)'
        },
        {
            'query': 'машинное AND обучение',
            'description': 'Russian AND query'
        },
        {
            'query': 'python AND NOT javascript',
            'description': 'NOT query (exclude term)'
        },
        {
            'query': 'веб AND разработка OR программирование',
            'description': 'Complex query with multiple operators'
        }
    ]

    print("=" * 70)
    print("BOOLEAN SEARCH TEST QUERIES")
    print("=" * 70)
    print()

    for i, test_case in enumerate(test_cases, 1):
        query = test_case['query']
        description = test_case['description']

        print(f"Query {i}: '{query}'")
        print(f"Description: {description}")
        print()

        results, total_count, elapsed_time = search(query)

        print(f"Results found: {total_count}")
        print(f"Total time: {elapsed_time:.3f} seconds")
        if len(results) > 0:
            print(f"Top-{len(results)}: {results}")
        else:
            print("No documents found.")
        print()
        print("-" * 70)
        print()

    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

if __name__ == '__main__':
    test_queries()