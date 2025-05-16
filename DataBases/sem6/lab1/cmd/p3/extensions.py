import os
import sys
import logging
from contextlib import suppress
from contextlib import contextmanager

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from internal.db.store import Store

@contextmanager
def timeout_context(seconds):
    yield

def main():
    logging.basicConfig(level=logging.INFO)
    store = Store()
    try:
        print("\n=== Fuzzy Search Manufacturers ===")
        with timeout_context(10):
            ms = store.fuzzy_search_manufacturers("deb", thresh=0.2, limit=20)
            for m in ms:
                print(f"ID={m['manufacturer_id']} Name={m['name']}")

        print("\n=== Fuzzy Search Products ===")
        with timeout_context(10):
            ps = store.fuzzy_search_products("diskwasher", thresh=0.3, limit=20)
            for p in ps:
                print(f"ID={p['product_id']} Category={p['category']} Price={p['price']:.2f}")

        with timeout_context(1000):
            store.encrypt_random_users("my-secret-key", total=20000)

    finally:
        store.close()

if __name__ == "__main__":
    main()