import sys
import os
import subprocess
from pymongo import MongoClient

def main():
    client = MongoClient('mongodb://localhost:27017/')
    db = client['search_engine']
    collection = db['documents']
    
    total_docs = collection.count_documents({})
    print(f"Found {total_docs} documents in MongoDB")
    
    if total_docs == 0:
        print("ERROR: No documents found in MongoDB!")
        return 1
    
    output_file = '../data/corpus_text.txt'
    os.makedirs('../data', exist_ok=True)
    
    doc_count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in collection.find().batch_size(100):
            html = doc.get('html', '')
            import re
            text = re.sub(r'<[^>]+>', ' ', html)
            text = re.sub(r'\s+', ' ', text).strip()
            
            if text and len(text) > 50:
                text = text.replace('\n', ' ').replace('\r', ' ')
                f.write(text + '\n---DOCUMENT---\n')
                doc_count += 1
                
                if doc_count % 1000 == 0:
                    print(f"Extracted {doc_count}/{total_docs} documents...")
    
    print(f"\nExtracted {doc_count} documents to {output_file}")
    
    if doc_count == 0:
        print("ERROR: No text extracted!")
        return 1
    
    print("Building search index...")
    result = subprocess.run([
        '../build/stemmer.exe', 'index', 
        '../data/corpus_text.txt',
        '../data/search_index.txt'
    ], capture_output=True, text=True, encoding='utf-8')
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode != 0:
        print(f"ERROR: Index build failed with code {result.returncode}")
        return 1
    
    # Проверяем размер индекса
    if os.path.exists('../data/search_index.txt'):
        size_mb = os.path.getsize('../data/search_index.txt') / (1024 * 1024)
        print(f"\nIndex file size: {size_mb:.2f} MB")
        if size_mb < 1:
            print("WARNING: Index file is suspiciously small!")
    else:
        print("ERROR: Index file was not created!")
        return 1
    
    print("\nStemming complete!")
    return 0

if __name__ == '__main__':
    sys.exit(main())