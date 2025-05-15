import time
import logging

class Store:
    def __init__(self, conn):
        self.conn = conn

    def close(self):
        self.conn.close()

    def create_indexes(self):
        logging.info("Создание индексов...")
        with self.conn.cursor() as cur:
            cur.execute("SET maintenance_work_mem = '1GB'")
            cur.execute("SET max_parallel_maintenance_workers = 4")

            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_products_price ON products(price);",
                "CREATE INDEX IF NOT EXISTS idx_orders_brin_date ON orders USING BRIN(order_date);",
                "CREATE INDEX IF NOT EXISTS idx_users_fullname_gin ON users USING GIN(to_tsvector('english', full_name));"
            ]
            for stmt in indexes:
                try:
                    cur.execute(stmt)
                except Exception as e:
                    logging.error(f"Ошибка при создании индекса: {e}")
                    raise
            self.conn.commit()
        logging.info("Индексы успешно созданы.")

    def drop_indexes(self):
        logging.info("Удаление индексов...")
        with self.conn.cursor() as cur:
            queries = [
                "DROP INDEX IF EXISTS idx_products_price;",
                "DROP INDEX IF EXISTS idx_orders_brin_date;",
                "DROP INDEX IF EXISTS idx_users_fullname_gin;",
            ]
            for q in queries:
                try:
                    cur.execute(q)
                except Exception as e:
                    logging.error(f"Ошибка при удалении: {e}")
                    raise
            self.conn.commit()
        logging.info("Индексы удалены.")

    def benchmark(self, label, use_index=True):
        if use_index:
            fullname_query = """
                SELECT COUNT(*) FROM (
                    SELECT * FROM users
                    WHERE full_name IS NOT NULL
                    AND to_tsvector('english', full_name) @@ plainto_tsquery('english', 'Anonymous')
                    LIMIT 250000
                ) AS limited;
            """
        else:
            fullname_query = """
                SELECT COUNT(*) FROM (
                    SELECT full_name FROM users
                    WHERE full_name IS NOT NULL
                    AND length(full_name) < 1000
                    LIMIT 250000
                ) AS limited
                WHERE to_tsvector('english', full_name) @@ plainto_tsquery('english', 'Anonymous');
            """

        tests = [
            ("Price Range", "SELECT COUNT(*) FROM products WHERE price BETWEEN 250 AND 500"),
            ("Date Range", "SELECT COUNT(*) FROM orders WHERE order_date BETWEEN '2019-11-05' AND '2019-11-06'"),
            ("FullName GIN Search", fullname_query)
        ]

        with self.conn.cursor() as cur:
            if not use_index:
                cur.execute("SET enable_indexscan = OFF;")
                cur.execute("SET enable_bitmapscan = OFF;")
                cur.execute("SET enable_seqscan = ON;")
                logging.info("Индексы временно отключены для теста")
            else:
                cur.execute("RESET enable_indexscan;")
                cur.execute("RESET enable_bitmapscan;")
                cur.execute("RESET enable_seqscan;")
                logging.info("Индексы будут использоваться")

            logging.info(f"=== Benchmark: {label} ===")

            for name, query in tests:
                for attempt in range(3):
                    try:
                        start = time.time()
                        cur.execute(query)
                        count = cur.fetchone()[0]
                        duration = time.time() - start
                        logging.info(f"{name}: {duration:.4f}s (rows={count})")
                        break
                    except Exception as e:
                        logging.warning(f"Ошибка в {name}, попытка #{attempt+1}: {e}")
                        time.sleep(2)