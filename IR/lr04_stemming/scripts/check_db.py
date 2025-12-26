from pymongo import MongoClient

try:
    client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=3000)
    client.server_info()
    
    db = client['search_engine']
    collection = db['documents']
    
    count = collection.count_documents({})
    print(f"MongoDB connected successfully")
    print(f"Database: search_engine")
    print(f"Documents found: {count}")
    
    if count > 0:
        # Показываем пример документа
        sample = collection.find_one()
        print(f"\nSample document:")
        print(f"  URL: {sample.get('url', 'N/A')}")
        print(f"  Source: {sample.get('source', 'N/A')}")
        print(f"  HTML size: {len(sample.get('html', ''))} bytes")
    else:
        print("\nNo documents in database!")
        print("Run crawler.py first (LR02)")
        
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    print("\nMake sure:")
    print("1. MongoDB is running")
    print("2. You ran crawler.py from LR02")