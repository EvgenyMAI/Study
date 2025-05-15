import time
import threading
from internal.db.db_config import init_db

class Store:
    def __init__(self):
        self.conn = init_db()

    def demo_non_repeatable_read(self):
        print(">>> Non‑Repeatable Read (READ COMMITTED) <<<")

        def tx1():
            with self.conn:
                with self.conn.cursor() as cur:
                    cur.execute("BEGIN ISOLATION LEVEL READ COMMITTED")

                    cur.execute("SELECT price FROM products WHERE product_id = %s", (1003461,))
                    price1 = cur.fetchone()[0]
                    print(f"Tx1 first read price = {price1:.2f}")

                    time.sleep(0.5)

                    cur.execute("SELECT price FROM products WHERE product_id = %s", (1003461,))
                    price2 = cur.fetchone()[0]
                    print(f"Tx1 second read price = {price2:.2f}")

                    cur.execute("COMMIT")

        def tx2():
            time.sleep(0.1)
            with self.conn:
                with self.conn.cursor() as cur:
                    cur.execute("BEGIN ISOLATION LEVEL READ COMMITTED")

                    cur.execute("UPDATE products SET price = price * 1.10 WHERE product_id = %s", (1003461,))
                    cur.execute("COMMIT")
                    print("Tx2 committed price update")

        thread1 = threading.Thread(target=tx1)
        thread2 = threading.Thread(target=tx2)
        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()

    def demo_phantom_read(self):
        print(">>> Phantom Read (READ COMMITTED) <<<")

        def tx1():
            with self.conn:
                with self.conn.cursor() as cur:
                    cur.execute("BEGIN ISOLATION LEVEL READ COMMITTED")

                    cur.execute("SELECT COUNT(*) FROM orders WHERE product_id = %s", (1003461,))
                    cnt1 = cur.fetchone()[0]
                    print(f"Tx1 first count = {cnt1}")

                    time.sleep(0.5)

                    cur.execute("SELECT COUNT(*) FROM orders WHERE product_id = %s", (1003461,))
                    cnt2 = cur.fetchone()[0]
                    print(f"Tx1 second count = {cnt2}")

                    cur.execute("COMMIT")

        def tx2():
            time.sleep(0.1)
            with self.conn:
                with self.conn.cursor() as cur:
                    cur.execute("BEGIN ISOLATION LEVEL READ COMMITTED")

                    cur.execute("""
                        INSERT INTO orders(user_id, product_id, order_date, status)
                        VALUES (%s, %s, NOW(), 'purchase')
                    """, (530496790, 1003461))

                    cur.execute("COMMIT")
                    print("Tx2 committed new order")

        thread1 = threading.Thread(target=tx1)
        thread2 = threading.Thread(target=tx2)
        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()