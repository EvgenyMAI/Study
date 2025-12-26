import sys
import os
import subprocess
from pymongo import MongoClient


def main():
    # Connect to MongoDB
    client = MongoClient('mongodb://localhost:27017/')
    db = client['search_engine']
    collection = db['documents']
    
    cursor = collection.find().batch_size(100)
    
    # Проверка на наличие документов (без выгрузки данных)
    total_docs = collection.count_documents({})
    print(f"Found {total_docs} documents in MongoDB")
    
    if total_docs == 0:
        print("ERROR: No documents found in MongoDB!")
        return 1
    
    # Extract text from documents
    output_file = '../data/corpus_text.txt'
    os.makedirs('../data', exist_ok=True)
    
    doc_count = 0
    import re
    
    print(f"Extracting text to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Потоковое чтение и запись
        for doc in cursor:
            html = doc.get('html', '')
            # Simple text extraction
            text = re.sub(r'<[^>]+>', ' ', html)
            text = re.sub(r'\s+', ' ', text).strip()
            
            if text and len(text) > 50:
                f.write(text + '\n')
                doc_count += 1
                
                # Опционально: вывод прогресса
                if doc_count % 1000 == 0:
                    print(f"Processed {doc_count} documents...", end='\r')
    
    print(f"\nExtracted {doc_count} documents to {output_file}")
    
    # Build index
    print("\nBuilding boolean index...")
    result = subprocess.run([
        '../build/boolean_index.exe', 'build', 
        '../data/corpus_text.txt',
        '../data/boolean_index.txt'
    ], capture_output=True, text=True, encoding='utf-8')
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode == 0:
        print("\nIndex built successfully!")
    else:
        print(f"Error building index (return code: {result.returncode})")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())