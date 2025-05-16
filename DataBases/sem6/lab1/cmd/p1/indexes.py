import os
import sys
import logging
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from internal.db.store import Store

def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="Benchmark-Index", help="режимы: Create-Noindexes, Drop-Indexes, Benchmark-Index, Benchmark-Noindex")
    args = parser.parse_args()

    store = Store()
    try:
        if args.mode == "Create-Indexes":
            print("=== Create-Indexes ===")
            store.create_indexes()
        elif args.mode == "Drop-Indexes":
            print("=== Drop-Indexes ===")
            store.drop_indexes()
        elif args.mode == "Benchmark-Index":
            print("=== Benchmark-Index ===")
            store.benchmark("с индексами", use_index=True)
        elif args.mode == "Benchmark-Noindex":
            print("=== Benchmark-Noindex ===")
            store.benchmark("без индексов", use_index=False)
        else:
            raise ValueError(f"Неизвестный режим: {args.mode}")
    finally:
        store.close()

if __name__ == "__main__":
    main()