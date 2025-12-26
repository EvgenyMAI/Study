from flask import Flask, render_template, request
import subprocess
import os

app = Flask(__name__)

# Путь к индексу и исполняемому файлу
INDEX_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'boolean_index.txt')
EXE_PATH = os.path.join(os.path.dirname(__file__), '..', 'build', 'boolean_search.exe')

def search_query(query):
    """Выполнить поисковый запрос"""
    if not query or len(query.strip()) == 0:
        return []
    
    try:
        # Запускаем процесс для каждого запроса отдельно
        result = subprocess.run(
            [EXE_PATH, INDEX_PATH, query],
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=30
        )
        
        lines = result.stdout.strip().split('\n')
        if len(lines) < 1:
            return []
        
        num_results = int(lines[0])
        doc_ids = [int(lines[i+1]) for i in range(min(num_results, len(lines)-1))]
        return doc_ids
    except subprocess.TimeoutExpired:
        print(f"Search timeout for query: {query}")
        return []
    except Exception as e:
        print(f"Search error: {e}")
        return []

@app.route('/')
def index():
    """Главная страница с формой поиска"""
    return render_template('index.html')

@app.route('/search')
def search():
    """Страница результатов поиска"""
    query = request.args.get('q', '')
    
    if not query:
        return render_template('index.html', error="Пустой запрос")
    
    print(f"Searching for: {query}")
    
    # Выполнить поиск
    doc_ids = search_query(query)
    
    print(f"Found: {len(doc_ids)} documents")
    
    # Получить информацию о документах из MongoDB (опционально)
    # Для простоты показываем только ID
    results = [{'doc_id': doc_id, 'title': f'Документ {doc_id}'} for doc_id in doc_ids[:50]]
    
    return render_template('results.html', 
                         query=query, 
                         results=results, 
                         total=len(doc_ids))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)