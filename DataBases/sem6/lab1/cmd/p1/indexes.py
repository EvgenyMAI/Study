import os
import sys
import logging
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from internal.db.store import Store

def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="benchmark-index", help="режим: create-noindexes, drop-indexes, benchmark-index, benchmark-noindex")
    args = parser.parse_args()

    store = Store()
    try:
        if args.mode == "create-indexes":
            store.create_indexes()
        elif args.mode == "drop-indexes":
            store.drop_indexes()
        elif args.mode == "benchmark-index":
            store.benchmark("с индексами", use_index=True)
        elif args.mode == "benchmark-noindex":
            store.benchmark("без индексов", use_index=False)
        else:
            raise ValueError(f"Неизвестный режим: {args.mode}")
    finally:
        store.close()

if __name__ == "__main__":
    main()