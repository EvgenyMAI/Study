from psycopg2.extensions import ISOLATION_LEVEL_READ_COMMITTED, ISOLATION_LEVEL_REPEATABLE_READ, ISOLATION_LEVEL_SERIALIZABLE

ISOLATION_LEVELS = {
    "READ COMMITTED": ISOLATION_LEVEL_READ_COMMITTED,
    "REPEATABLE READ": ISOLATION_LEVEL_REPEATABLE_READ,
    "SERIALIZABLE": ISOLATION_LEVEL_SERIALIZABLE,
}

class Store:
    def __init__(self, conn):
        self.conn = conn

    def place_order(self, tx, user_id, product_id, qty):
        with tx.cursor() as cur:
            # 1. Получаем цену и остаток
            cur.execute("""
                SELECT price, stock_quantity
                FROM products
                WHERE product_id = %s
                FOR UPDATE
            """, (product_id,))
            row = cur.fetchone()
            if not row:
                raise Exception("Product not found")
            price, stock = row
            if stock < qty:
                raise Exception("Not enough stock")

            # 2. Проверяем баланс пользователя
            cur.execute("""
                SELECT balance
                FROM users
                WHERE user_id = %s
                FOR UPDATE
            """, (user_id,))
            row = cur.fetchone()
            if not row:
                raise Exception("User not found")
            balance = row[0]
            total = price * qty
            if balance < total:
                raise Exception("Not enough funds")

            # 3. Списываем у пользователя
            cur.execute("""
                UPDATE users
                SET balance = balance - %s
                WHERE user_id = %s
            """, (total, user_id))

            # 4. Уменьшаем остаток на складе
            cur.execute("""
                UPDATE products
                SET stock_quantity = stock_quantity - %s
                WHERE product_id = %s
            """, (qty, product_id))

            # 5. Создаем заказ
            cur.execute("""
                INSERT INTO orders(user_id, product_id, order_date, status)
                VALUES(%s, %s, NOW(), 'purchase')
            """, (user_id, product_id))

    def restock_product(self, tx, product_id, qty):
        with tx.cursor() as cur:
            cur.execute("""
                UPDATE products
                SET stock_quantity = stock_quantity + %s
                WHERE product_id = %s
            """, (qty, product_id))

    def transfer_funds(self, tx, from_id, to_id, amount):
        with tx.cursor() as cur:
            # Блокируем в порядке возрастания ID
            for uid in sorted([from_id, to_id]):
                cur.execute("SELECT 1 FROM users WHERE user_id = %s FOR UPDATE", (uid,))

            # Списание
            cur.execute("""
                UPDATE users SET balance = balance - %s
                WHERE user_id = %s
            """, (amount, from_id))

            # Зачисление
            cur.execute("""
                UPDATE users SET balance = balance + %s
                WHERE user_id = %s
            """, (amount, to_id))

    def perform_transaction(self, isolation_level="READ COMMITTED"):
        level = ISOLATION_LEVELS.get(isolation_level.upper())
        if level is None:
            raise ValueError(f"Unsupported isolation level: {isolation_level}")

        self.conn.set_isolation_level(level)
        tx = self.conn.cursor()
        try:
            with self.conn:
                # Выполняем заказ
                self.place_order(self.conn, 530496790, 5000088, 2)

                # Пополняем склад
                self.restock_product(self.conn, 5000088, 10)

            # После фиксации — получаем результат
            with self.conn.cursor() as cur:
                cur.execute("SELECT balance FROM users WHERE user_id = %s", (530496790,))
                balance = cur.fetchone()[0]

                cur.execute("SELECT stock_quantity FROM products WHERE product_id = %s", (5000088,))
                stock = cur.fetchone()[0]

            print(f"User 530496790 balance: {balance:.2f}")
            print(f"Product 5000088 quantity: {stock}")
        except Exception as e:
            self.conn.rollback()
            raise e

    def perform_transfer(self, isolation_level, from_id, to_id, amount):
        level = ISOLATION_LEVELS.get(isolation_level.upper())
        if level is None:
            raise ValueError(f"Unsupported isolation level: {isolation_level}")

        self.conn.set_isolation_level(level)
        try:
            with self.conn:
                self.transfer_funds(self.conn, from_id, to_id, amount)

            # Показать балансы
            with self.conn.cursor() as cur:
                cur.execute("SELECT balance FROM users WHERE user_id = %s", (from_id,))
                from_balance = cur.fetchone()[0]
                cur.execute("SELECT balance FROM users WHERE user_id = %s", (to_id,))
                to_balance = cur.fetchone()[0]

            print(f"User {from_id} balance: {from_balance:.2f}")
            print(f"User {to_id} balance: {to_balance:.2f}")
        except Exception as e:
            self.conn.rollback()
            raise e