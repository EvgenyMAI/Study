from internal.db.db_config import init_db
from internal.db.store_1 import Store as Store_1_1
from internal.db.store_2 import Store as Store_1_2
from internal.db.anomalyes_2 import Store as Anomalies
from internal.db.extensions_3 import Extensions

class Store:
    def __init__(self):
        self.conn = init_db()

        self.indexing = Store_1_1(self.conn)
        self.transactions = Store_1_2(self.conn)
        self.anomalies = Anomalies()
        self.extensions = Extensions(self.conn)

    def close(self):
        self.conn.close()

    # Прокси-методы
    def create_indexes(self):
        self.indexing.create_indexes()

    def drop_indexes(self):
        self.indexing.drop_indexes()

    def benchmark(self, label, use_index=True):
        self.indexing.benchmark(label, use_index)

    def perform_transaction(self, isolation_level="READ COMMITTED"):
        self.transactions.perform_transaction(isolation_level)

    def perform_transfer(self, isolation_level, from_id, to_id, amount):
        self.transactions.perform_transfer(isolation_level, from_id, to_id, amount)

    def demo_non_repeatable_read(self):
        self.anomalies.demo_non_repeatable_read()

    def demo_phantom_read(self):
        self.anomalies.demo_phantom_read()

    def fuzzy_search_manufacturers(self, term, thresh=0.3, limit=10):
        return self.extensions.fuzzy_search_manufacturers(term, thresh, limit)

    def fuzzy_search_products(self, term, thresh=0.3, limit=10):
        return self.extensions.fuzzy_search_products(term, thresh, limit)

    def encrypt_random_users(self, key, total=100000):
        self.extensions.encrypt_random_users(key, total)

    def get_decrypted_email_by_id(self, user_id, key):
        return self.extensions.get_decrypted_email_by_id(user_id, key)
