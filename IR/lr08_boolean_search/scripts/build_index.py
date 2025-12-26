import sys
import os
import shutil

def main():
    print("Building boolean index for search...")
    print()
    
    # Определяем абсолютные пути
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lr08_root = os.path.dirname(script_dir)
    lr07_root = os.path.join(os.path.dirname(lr08_root), 'lr07_boolean_index')
    
    lr07_index = os.path.join(lr07_root, 'data', 'boolean_index.txt')
    lr08_index = os.path.join(lr08_root, 'data', 'boolean_index.txt')
    
    print(f"LR07 index path: {lr07_index}")
    print(f"LR08 index path: {lr08_index}")
    print()
    
    # Проверяем есть ли индекс в ЛР7
    if os.path.exists(lr07_index):
        print("Found index in LR07")
        print("Copying to LR08...")
        
        os.makedirs(os.path.dirname(lr08_index), exist_ok=True)
        shutil.copy2(lr07_index, lr08_index)
        
        print("Index copied successfully!")
        
        # Проверяем размер
        size_mb = os.path.getsize(lr08_index) / (1024 * 1024)
        print(f"Index size: {size_mb:.2f} MB")
        
    else:
        print("ERROR: Index not found in LR07!")
        print(f"Expected: {lr07_index}")
        print()
        print("Please build the index first:")
        print("cd lr07_boolean_index")
        print("run_build_index.bat")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())