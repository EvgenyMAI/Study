import logging
from internal.db.store import Store

def main():
    logging.basicConfig(level=logging.INFO)
    store = Store()
    try:
        print("=== SERIALIZABLE ===")
        store.perform_transaction("SERIALIZABLE")

        print("=== REPEATABLE READ ===")
        store.perform_transfer("REPEATABLE READ", 520088904, 561587266, 10.0)

        print("\n=== Non‑Repeatable Read ===")
        store.demo_non_repeatable_read()

        print("\n=== Phantom Read ===")
        store.demo_phantom_read()
    finally:
        store.close()

if __name__ == "__main__":
    main()