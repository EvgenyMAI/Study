import os
import csv
import random
import logging
import tempfile
import psycopg2
from datetime import datetime
from dotenv import load_dotenv
from psycopg2.extras import execute_values
from typing import Dict, List, Set, Tuple, Optional

# Configuration
class Config:
    TARGET_USERS = 5_000_000
    BATCH_SETTINGS = {
        'users': 200_000,
        'orders': 300_000
    }
    DATA_RANGES = {
        'price': (100, 10_000),
        'stock': (5, 1000),
        'warranty': (6, 60),
        'balance': (100.0, 15_000.0)
    }
    STATUS_MAPPING = {
        'purchase': 'completed',
        'cart': 'pending',
        'view': None
    }

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('db_population.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('DB_Populator')

class DatabaseManager:
    """Manages database connections and operations"""
    
    def __init__(self):
        self.conn = None
        self._staging_tables_created = False
        
    def connect(self) -> None:
        """Establish database connection"""
        try:
            load_dotenv('./environment/psql_db.env')
            self.conn = psycopg2.connect(
                host=os.getenv('POSTGRES_HOST'),
                port=os.getenv('POSTGRES_PORT'),
                dbname=os.getenv('POSTGRES_DB'),
                user=os.getenv('POSTGRES_USER'),
                password=os.getenv('POSTGRES_PASSWORD'),
                connect_timeout=15
            )
            self.conn.autocommit = False
            logger.info("Database connection established")
        except psycopg2.Error as e:
            logger.error(f"Connection failed: {e}")
            raise

    def close(self) -> None:
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("Database connection closed")

    def table_exists(self, table_name: str) -> bool:
        """Check if table exists in database"""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables 
                    WHERE table_name = %s
                )
            """, (table_name,))
            return cur.fetchone()[0]

    def create_staging_tables(self) -> None:
        """Create UNLOGGED staging tables"""
        if self._staging_tables_created:
            return

        queries = [
            """CREATE UNLOGGED TABLE IF NOT EXISTS staging_brands (
                brand_name TEXT PRIMARY KEY
            )""",
            """CREATE UNLOGGED TABLE IF NOT EXISTS staging_products (
                product_id BIGINT,
                brand TEXT,
                price INTEGER,
                category TEXT,
                inventory INTEGER,
                warranty INTEGER
            )""",
            """CREATE UNLOGGED TABLE IF NOT EXISTS staging_transactions (
                customer_id BIGINT,
                item_id BIGINT,
                transaction_time TIMESTAMP,
                transaction_type TEXT
            )""",
            """CREATE INDEX IF NOT EXISTS idx_staging_brand ON staging_products(brand)""",
            """CREATE INDEX IF NOT EXISTS idx_staging_product_id ON staging_products(product_id)"""
        ]

        with self.conn.cursor() as cur:
            for query in queries:
                try:
                    cur.execute(query)
                except psycopg2.Error as e:
                    logger.error(f"Error executing query: {query}\nError: {e}")
                    raise
        self.conn.commit()
        self._staging_tables_created = True
        logger.info("Staging tables created")

    def clear_staging_tables(self) -> None:
        """Clear data from staging tables"""
        if not self._staging_tables_created:
            return

        tables = [
            'staging_brands',
            'staging_products',
            'staging_transactions'
        ]

        with self.conn.cursor() as cur:
            for table in tables:
                cur.execute(f"TRUNCATE TABLE {table}")
        self.conn.commit()
        logger.info("Staging tables cleared")

    def cleanup_staging_tables(self):
        """Remove staging tables after use"""
        tables = [
            'staging_brands',
            'staging_products',
            'staging_transactions'
        ]
        
        with self.conn.cursor() as cur:
            for table in tables:
                try:
                    cur.execute(f"DROP TABLE IF EXISTS {table}")
                    logger.info(f"Dropped table {table}")
                except psycopg2.Error as e:
                    logger.error(f"Error dropping table {table}: {e}")
        self.conn.commit()

    def load_from_csv(self, table_name: str, file_path: str) -> None:
        """Load data from CSV to staging table"""
        if not self.table_exists(table_name):
            raise ValueError(f"Table {table_name} does not exist")

        with open(file_path, 'r', encoding='utf-8') as f:
            with self.conn.cursor() as cur:
                try:
                    cur.copy_expert(
                        f"COPY {table_name} FROM STDIN WITH CSV HEADER",
                        f
                    )
                    self.conn.commit()
                    logger.info(f"Data loaded to {table_name}")
                except psycopg2.Error as e:
                    self.conn.rollback()
                    logger.error(f"Failed to load data to {table_name}: {e}")
                    raise

class DataProcessor:
    """Processes and transforms raw data"""
    
    @staticmethod
    def parse_user_data(file_path: str) -> List[Tuple[str, str]]:
        """Parse user data from CSV"""
        user_profiles = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                reader = csv.DictReader((line.replace('\0', '') for line in file))
                for record in reader:
                    user_profiles.append((
                        record.get('email', f"user_{random.randint(1_000_000, 9_999_999)}@example.com"),
                        record.get('full_name', 'Anonymous User')
                    ))
            logger.info(f"Loaded {len(user_profiles)} user profiles")
            return user_profiles
        except Exception as e:
            logger.error(f"Failed to parse user data: {e}")
            raise

    @staticmethod
    def generate_product_metadata(raw_category: str) -> Tuple[str, int, int, int]:
        """Generate product metadata with realistic distributions"""
        category_parts = raw_category.split('.') if raw_category else []
        category = category_parts[-1] if category_parts else 'uncategorized'
        
        # Логнормальное распределение для цены
        price = int(100 * (2 ** random.normalvariate(3, 0.8)))
        price = max(100, min(price, 50_000))
        
        # Отрицательное биномиальное распределение для запасов
        stock = random.randint(0, 1000) if random.random() < 0.8 else random.randint(1000, 10_000)
        
        # Гарантия - чаще 12-24 месяца
        warranty = random.choices(
            [12, 24, 6, 36, 60],
            weights=[0.4, 0.4, 0.1, 0.05, 0.05]
        )[0]
        
        return category, price, stock, warranty

    @staticmethod
    def parse_timestamp(event_time: str) -> Optional[datetime]:
        """Parse and validate event timestamp with multiple formats"""
        formats = [
            '%Y-%m-%d %H:%M:%S %Z',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M',
            '%Y-%m-%d'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(event_time, fmt)
            except (ValueError, TypeError):
                continue
        return None

class DataGenerator:
    """Generates and processes dataset files"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.temp_dir = tempfile.mkdtemp(prefix='db_population_')
        
    def prepare_data_files(self, source_file: str, user_profiles: List[Tuple[str, str]]) -> Dict[str, str]:
        """Prepare all data files from source dataset"""
        try:
            file_paths = {
                'brands': os.path.join(self.temp_dir, 'brands.csv'),
                'products': os.path.join(self.temp_dir, 'products.csv'),
                'users': os.path.join(self.temp_dir, 'users.csv'),
                'transactions': os.path.join(self.temp_dir, 'transactions.csv')
            }
            
            self._write_headers(file_paths)
            stats = self._process_source_data(source_file, user_profiles, file_paths)
            self._generate_missing_users(file_paths['users'], user_profiles, stats['users'])
            
            logger.info(f"Data files prepared. Stats: {stats}")
            return file_paths
            
        except Exception as e:
            logger.error(f"Failed to prepare data files: {e}")
            raise

    def _write_headers(self, file_paths: Dict[str, str]) -> None:
        """Write CSV headers for all files"""
        headers = {
            'brands': ['brand_name'],
            'products': ['product_id', 'brand', 'price', 'category', 'inventory', 'warranty'],
            'users': ['user_id', 'email', 'balance', 'full_name'],
            'transactions': ['customer_id', 'item_id', 'transaction_time', 'transaction_type']
        }
        
        for key, path in file_paths.items():
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers[key])

    def _process_source_data(self, source_file: str, user_profiles: List[Tuple[str, str]],
                           file_paths: Dict[str, str]) -> Dict[str, int]:
        """Process source data file and generate output files"""
        stats = {
            'rows': 0,
            'brands': set(),
            'users': set(),
            'products': set(),
            'transactions': 0
        }
        
        try:
            with open(source_file, 'r', encoding='utf-8', errors='replace') as src_file:
                reader = csv.DictReader((line.replace('\0', '') for line in src_file))
                
                for record in reader:
                    stats['rows'] += 1
                    if stats['rows'] % 1_000_000 == 0:
                        logger.info(f"Processed {stats['rows']} rows")
                    
                    # Process brand
                    brand = record.get('brand')
                    if not brand:
                        continue
                    
                    self._process_brand(brand, file_paths['brands'], stats['brands'])
                    
                    # Process user
                    user_id = self._parse_id(record.get('user_id'))
                    if user_id and user_id not in stats['users']:
                        self._process_user(user_id, user_profiles, file_paths['users'], stats['users'])
                    
                    # Process product
                    product_id = self._parse_id(record.get('product_id'))
                    if product_id and product_id not in stats['products']:
                        self._process_product(
                            product_id, brand, record.get('category_code', ''),
                            file_paths['products'], stats['products']
                        )
                    
                    # Process transaction
                    self._process_transaction(
                        user_id, product_id, record.get('event_type'),
                        record.get('event_time', ''), file_paths['transactions'], stats
                    )
            
            return stats
            
        except Exception as e:
            logger.error(f"Error processing source data: {e}")
            raise

    def _process_brand(self, brand: str, file_path: str, seen_brands: Set[str]) -> None:
        """Process brand data"""
        if brand not in seen_brands:
            with open(file_path, 'a', newline='', encoding='utf-8') as f:
                f.write(f"{brand}\n")
            seen_brands.add(brand)

    def _process_user(self, user_id: int, user_profiles: List[Tuple[str, str]],
                     file_path: str, seen_users: Set[int]) -> None:
        """Process user data"""
        email, full_name = random.choice(user_profiles)
        balance = round(random.uniform(*Config.DATA_RANGES['balance']), 2)
        with open(file_path, 'a', newline='', encoding='utf-8') as f:
            f.write(f"{user_id},{email},{balance},{full_name}\n")
        seen_users.add(user_id)

    def _process_product(self, product_id: int, brand: str, category_code: str,
                        file_path: str, seen_products: Set[int]) -> None:
        """Process product data"""
        category, price, stock, warranty = DataProcessor.generate_product_metadata(category_code)
        with open(file_path, 'a', newline='', encoding='utf-8') as f:
            f.write(f"{product_id},{brand},{price},{category},{stock},{warranty}\n")
        seen_products.add(product_id)

    def _process_transaction(self, user_id: int, product_id: int, event_type: str,
                            event_time: str, file_path: str, stats: Dict[str, int]) -> None:
        """Process transaction data"""
        status = Config.STATUS_MAPPING.get(event_type)
        if status is None:
            return
            
        timestamp = DataProcessor.parse_timestamp(event_time)
        if not timestamp:
            return
            
        with open(file_path, 'a', newline='', encoding='utf-8') as f:
            f.write(f"{user_id},{product_id},{timestamp},{status}\n")
        stats['transactions'] += 1

    def _generate_missing_users(self, file_path: str, profiles: List[Tuple[str, str]],
                              existing_users: Set[int]) -> None:
        """Generate additional users to reach target count"""
        needed = Config.TARGET_USERS - len(existing_users)
        if needed <= 0:
            return
            
        max_id = max(existing_users) if existing_users else 0
        logger.info(f"Generating {needed} additional users")
        
        with open(file_path, 'a', newline='', encoding='utf-8') as f:
            for i in range(1, needed + 1):
                user_id = max_id + i
                email, full_name = profiles[i % len(profiles)]
                balance = round(random.uniform(*Config.DATA_RANGES['balance']), 2)
                f.write(f"{user_id},{email},{balance},{full_name}\n")

    def _parse_id(self, id_str: str) -> Optional[int]:
        """Parse ID from string with validation"""
        try:
            return int(id_str) if id_str else None
        except ValueError:
            return None

class DataLoader:
    """Handles data loading to database"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        
    def load_all_data(self, data_files: Dict[str, str]) -> None:
        """Load all data from files to database"""
        try:
            # Load staging data
            self.db.load_from_csv('staging_brands', data_files['brands'])
            self.db.load_from_csv('staging_products', data_files['products'])
            self.db.load_from_csv('staging_transactions', data_files['transactions'])
            
            # Transfer data to main tables
            self._transfer_brands()
            self._transfer_products()
            
            # Load users and orders
            self._load_users(data_files['users'])
            self._load_orders(data_files['transactions'])

            self.db.cleanup_staging_tables()
            
            logger.info("All data loaded successfully")
            
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            raise

    def _transfer_brands(self) -> None:
        """Transfer brands data from staging to main table"""
        with self.db.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO manufacturers(name)
                SELECT brand_name FROM staging_brands
                ON CONFLICT (name) DO NOTHING
            """)
            self.db.conn.commit()
            logger.info("Brands data transferred")

    def _transfer_products(self) -> None:
        """Transfer products data from staging to main table"""
        with self.db.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO products(
                    product_id, manufacturer_id, price, 
                    category, stock_quantity, warranty_period
                )
                SELECT 
                    sp.product_id, 
                    m.manufacturer_id, 
                    sp.price, 
                    sp.category, 
                    sp.inventory, 
                    sp.warranty
                FROM staging_products sp
                JOIN manufacturers m ON m.name = sp.brand
                ON CONFLICT (product_id) DO NOTHING
            """)
            self.db.conn.commit()
            logger.info("Products data transferred")

    def _load_users(self, file_path: str) -> None:
        """Load users data in batches"""
        logger.info(f"Loading users in batches of {Config.BATCH_SETTINGS['users']}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            total = sum(1 for _ in f) - 1
            f.seek(0)
            next(f)  # Skip header
            
            batch = []
            loaded = 0
            cur = self.db.conn.cursor()
            
            for line in f:
                parts = line.strip().split(',')
                if len(parts) != 4:
                    continue
                    
                batch.append((
                    int(parts[0]),  # user_id
                    parts[1],       # email
                    float(parts[2]), # balance
                    parts[3]        # full_name
                ))
                
                if len(batch) >= Config.BATCH_SETTINGS['users']:
                    execute_values(
                        cur,
                        """INSERT INTO users(
                            user_id, email, balance, full_name
                        ) VALUES %s ON CONFLICT DO NOTHING""",
                        batch
                    )
                    self.db.conn.commit()
                    loaded += len(batch)
                    logger.info(f"Loaded {loaded}/{total} users")
                    batch = []
            
            if batch:
                execute_values(
                    cur,
                    """INSERT INTO users(
                        user_id, email, balance, full_name
                    ) VALUES %s ON CONFLICT DO NOTHING""",
                    batch
                )
                self.db.conn.commit()
                loaded += len(batch)
                logger.info(f"Loaded {loaded}/{total} users (final batch)")
            
            cur.close()

    def _load_orders(self, file_path: str) -> None:
        """Load orders data in batches"""
        logger.info(f"Loading orders in batches of {Config.BATCH_SETTINGS['orders']}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            total = sum(1 for _ in f) - 1
            f.seek(0)
            next(f)  # Skip header
            
            batch = []
            loaded = 0
            cur = self.db.conn.cursor()
            
            for line in f:
                parts = line.strip().split(',')
                if len(parts) != 4:
                    continue
                    
                batch.append((
                    int(parts[0]),     # user_id
                    int(parts[1]),     # product_id
                    parts[2],          # order_date
                    parts[3]           # status
                ))
                
                if len(batch) >= Config.BATCH_SETTINGS['orders']:
                    execute_values(
                        cur,
                        """INSERT INTO orders(
                            user_id, product_id, order_date, status
                        ) VALUES %s""",
                        batch
                    )
                    self.db.conn.commit()
                    loaded += len(batch)
                    logger.info(f"Loaded {loaded}/{total} orders")
                    batch = []
            
            if batch:
                execute_values(
                    cur,
                    """INSERT INTO orders(
                        user_id, product_id, order_date, status
                    ) VALUES %s""",
                    batch
                )
                self.db.conn.commit()
                loaded += len(batch)
                logger.info(f"Loaded {loaded}/{total} orders (final batch)")
            
            cur.close()

def main():
    """Main execution flow"""
    db = DatabaseManager()
    try:
        db.connect()
        
        # Initialize staging environment
        db.create_staging_tables()
        db.clear_staging_tables()
         
        # Process data
        processor = DataProcessor()
        user_profiles = processor.parse_user_data('./dataset/generated_users.csv')
        
        generator = DataGenerator(db)
        data_files = generator.prepare_data_files(
            './dataset/downloaded_dataset.csv',
            user_profiles
        )
        
        # Load data to database
        loader = DataLoader(db)
        loader.load_all_data(data_files)
        
        logger.info("Database population completed successfully")
        
    except Exception as e:
        logger.critical(f"Process failed: {e}")
        raise
    finally:
        db.close()

if __name__ == '__main__':
    main()