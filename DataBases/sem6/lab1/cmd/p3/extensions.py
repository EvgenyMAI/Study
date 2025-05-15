import logging
from contextlib import suppress
from internal.db.store import Store
from contextlib import contextmanager


@contextmanager
def timeout_context(seconds):
    yield  # Заглушка — можно использовать `asyncio.Timeout` или `signal` в реальной реализации

def main():
    logging.basicConfig(level=logging.INFO)
    store = Store()
    try:
        print("\n=== Fuzzy Search Manufacturers ===")
        with timeout_context(5):
            ms = store.fuzzy_search_manufacturers("maliz", method="trgm", threshold=0.2, limit=10)
            for m in ms:
                print(f"ID={m.id} Name={m.name}")

        print("\n=== Fuzzy Search Products ===")
        with timeout_context(5):
            ps = store.fuzzy_search_products("smartphone", method="bigm", threshold=0.3, limit=5)
            for p in ps:
                print(f"ID={p.id} Category={p.category} Price={p.price:.2f}")

        with timeout_context(500):
            store.encrypt_random_users("my-secret-key", count=10000)

        with suppress(Exception):
            email = store.get_decrypted_email_by_id(581234831, "my-secret-key")
            print(f"Email пользователя 581234831: {email}")
    finally:
        store.close()

if __name__ == "__main__":
    main()