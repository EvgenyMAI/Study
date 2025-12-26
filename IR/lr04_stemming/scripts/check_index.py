import os

print("Checking index files...")
print()

corpus_file = '../data/corpus_text.txt'
index_file = '../data/search_index.txt'

if os.path.exists(corpus_file):
    size_mb = os.path.getsize(corpus_file) / (1024 * 1024)
    lines = 0
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            lines += 1
    print(f"Corpus file: {size_mb:.2f} MB, {lines} lines")
else:
    print("Corpus file not found!")

if os.path.exists(index_file):
    size_mb = os.path.getsize(index_file) / (1024 * 1024)
    lines = 0
    with open(index_file, 'r', encoding='utf-8') as f:
        for line in f:
            lines += 1
    print(f"Index file: {size_mb:.2f} MB, {lines} lines")
    
    # Читаем первые строки
    with open(index_file, 'r', encoding='utf-8') as f:
        num_docs = int(f.readline().strip())
        num_terms = int(f.readline().strip())
        print(f"Documents: {num_docs}")
        print(f"Terms: {num_terms}")
else:
    print("Index file not found!")

print()
print("Run this to rebuild:")
print(".\\run_stemming.bat")