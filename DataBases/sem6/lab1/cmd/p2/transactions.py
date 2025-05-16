import os
import sys
import logging
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from internal.db.store import Store

def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="Serializable", help="режимы: Serializable, Repeatable Read, Non‑Repeatable Read, Phantom Read")
    args = parser.parse_args()

    store = Store()
    try:
        if args.mode == "Serializable":
            print("=== Serializable ===")
            store.perform_transaction("SERIALIZABLE")
        elif args.mode == "Repeatable Read":
            print("=== Repeatable Read ===")
            store.perform_transfer("REPEATABLE READ", 520088904, 561587266, 10.0)
        elif args.mode == "Non‑Repeatable Read":
            print("\n=== Non‑Repeatable Read ===")
            store.demo_non_repeatable_read()
        elif args.mode == "Phantom Read":
            print("\n=== Phantom Read ===")
            store.demo_phantom_read()
        else:
            raise ValueError(f"Неизвестный режим: {args.mode}")
    finally:
        store.close()

if __name__ == "__main__":
    main()