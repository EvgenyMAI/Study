import logging
import psycopg2
import psycopg2.extras

class Extensions:
    def __init__(self, conn):
        self.conn = conn

    def ensure_fuzzy_extensions_and_indexes(self):
        with self.conn.cursor() as cur:
            stmts = [
                "CREATE EXTENSION IF NOT EXISTS pg_trgm",
                "CREATE EXTENSION IF NOT EXISTS pg_bigm",
                "CREATE EXTENSION IF NOT EXISTS pgcrypto",

                "CREATE INDEX IF NOT EXISTS idx_manufacturers_name_trgm ON manufacturers USING gin (name gin_trgm_ops)",
                "CREATE INDEX IF NOT EXISTS idx_manufacturers_name_bigm ON manufacturers USING gin (name gin_bigm_ops)",

                "CREATE INDEX IF NOT EXISTS idx_products_category_trgm ON products USING gin (category gin_trgm_ops)",
                "CREATE INDEX IF NOT EXISTS idx_products_category_bigm ON products USING gin (category gin_bigm_ops)",
            ]
            for i, stmt in enumerate(stmts, 1):
                try:
                    cur.execute(stmt)
                except Exception as e:
                    raise RuntimeError(f"Step {i} failed ({stmt}): {e}")
            self.conn.commit()

    def fuzzy_search_manufacturers(self, term, thresh=0.3, limit=10):
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            query = """
                SELECT manufacturer_id, name
                FROM manufacturers
                WHERE similarity(name, %s) > %s
                ORDER BY similarity(name, %s) DESC
                LIMIT %s
            """
            cur.execute(query, (term, thresh, term, limit))
            return cur.fetchall()

    def fuzzy_search_products(self, term, thresh=0.3, limit=10):
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            query = """
                SELECT product_id, category, price
                FROM products
                WHERE similarity(category, %s) > %s
                ORDER BY similarity(category, %s) DESC
                LIMIT %s
            """
            cur.execute(query, (term, thresh, term, limit))
            return cur.fetchall()

    def create_user_encrypted(self, user_id, email, name, balance, key):
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO users(user_id, email_enc, balance, name)
                VALUES (%s, pgp_sym_encrypt(%s, %s), %s, %s)
            """, (user_id, email, key, balance, name))
            self.conn.commit()

    def get_user_decrypted(self, user_id, key):
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT pgp_sym_decrypt(email_enc, %s)::text, name, balance
                FROM users
                WHERE user_id = %s
            """, (key, user_id))
            return cur.fetchone()

    def ensure_encrypted_users_table(self):
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS encrypted_users (
                    user_id BIGINT PRIMARY KEY,
                    encrypted_email BYTEA NOT NULL,
                    CONSTRAINT fk_user_id FOREIGN KEY (user_id)
                        REFERENCES users(user_id)
                        ON DELETE CASCADE
                )
            """)
            self.conn.commit()

    def encrypt_random_users(self, key, total=100000):
        batch_size = 10_000
        logging.info("Создание таблицы encrypted_users (если не существует)...")
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS encrypted_users (
                    user_id INT PRIMARY KEY,
                    email_enc BYTEA
                )
            """)
            self.conn.commit()

        logging.info(f"Начинаем шифрование {total} пользователей батчами по {batch_size}...")

        for offset in range(0, total, batch_size):
            logging.info(f"Шифруем пользователей: {offset+1} – {offset+batch_size}")
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO encrypted_users (user_id, email_enc)
                    SELECT user_id, pgp_sym_encrypt(email, %s)
                    FROM users
                    ORDER BY random()
                    LIMIT %s
                    ON CONFLICT (user_id) DO NOTHING
                """, (key, batch_size))
                self.conn.commit()

        logging.info("✅ Шифрование завершено")

    def get_decrypted_email_by_id(self, user_id, key):
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT pgp_sym_decrypt(email_enc, %s)::text
                FROM encrypted_users
                WHERE user_id = %s
            """, (key, user_id))
            row = cur.fetchone()
            if not row:
                raise ValueError(f"User {user_id} not found or email not decrypted")
            return row[0]