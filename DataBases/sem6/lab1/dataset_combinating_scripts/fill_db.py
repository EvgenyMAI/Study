import os
import csv
import random
import tempfile
from datetime import datetime
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values

TARGET_USER_COUNT = 5000000

# Load environment variables
def load_config(env_path='../environment/psql_db.env'):
    load_dotenv(env_path)
    return {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '28879'),
        'dbname': os.getenv('POSTGRES_DB'),
        'user': os.getenv('POSTGRES_USER'),
        'password': os.getenv('POSTGRES_PASSWORD'),
        'sslmode': 'disable'
    }

# Establish PostgreSQL connection
def init_db(config):
    conn = psycopg2.connect(**config)
    conn.autocommit = False
    return conn

# Set up temporary staging tables for non-user data
def setup_staging(cur):
    cur.execute("""
    CREATE TEMP TABLE staging_manufacturers(name TEXT);
    CREATE TEMP TABLE staging_products(
        product_id BIGINT,
        manufacturer_name TEXT,
        price INTEGER,
        category TEXT,
        stock_quantity INTEGER,
        warranty_period INTEGER
    );
    CREATE TEMP TABLE staging_orders(
        user_id BIGINT,
        product_id BIGINT,
        order_date TIMESTAMP,
        status TEXT
    );
    """)

# Load user info (email, full_name)
def load_user_infos(path):
    users = []
    with open(path, 'r', newline='', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader((line.replace('\x00', '') for line in f))
        for row in reader:
            users.append((row.get('email', ''), row.get('full_name', '')))
    return users

# Stream through big dataset, writing staging CSVs and synthetic users
def generate_staging_files(big_csv, user_infos, report_interval=1000000):
    tmp_dir = tempfile.mkdtemp()
    files = {
        'manufacturers': open(os.path.join(tmp_dir, 'manufacturers.csv'), 'w', newline='', encoding='utf-8'),
        'users': open(os.path.join(tmp_dir, 'users.csv'), 'w', newline='', encoding='utf-8'),
        'products': open(os.path.join(tmp_dir, 'products.csv'), 'w', newline='', encoding='utf-8'),
        'orders': open(os.path.join(tmp_dir, 'orders.csv'), 'w', newline='', encoding='utf-8')
    }
    writers = {k: csv.writer(v) for k, v in files.items()}

    seen_brands = set()
    seen_users = set()
    seen_products = set()
    row_count = 0

    # Write headers
    writers['manufacturers'].writerow(['name'])
    writers['users'].writerow(['user_id', 'email', 'balance', 'full_name'])
    writers['products'].writerow(['product_id', 'manufacturer_name', 'price', 'category', 'stock_quantity', 'warranty_period'])
    writers['orders'].writerow(['user_id', 'product_id', 'order_date', 'status'])

    with open(big_csv, 'r', newline='', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader((line.replace('\x00', '') for line in f))
        for row in reader:
            row_count += 1
            brand = row.get('brand')
            if not brand:
                continue
            try:
                user_id = int(row.get('user_id', '0'))
                prod_id = int(row.get('product_id', '0'))
            except ValueError:
                continue
            event_type = row.get('event_type', '')

            # Manufacturer
            if brand not in seen_brands:
                writers['manufacturers'].writerow([brand])
                seen_brands.add(brand)

            # User from dataset
            if user_id not in seen_users:
                email, full_name = random.choice(user_infos)
                balance = round(random.random() * 10000, 2)
                writers['users'].writerow([user_id, email, balance, full_name])
                seen_users.add(user_id)

            # Product
            if prod_id not in seen_products:
                try:
                    price = int(float(row.get('price', 0)))
                except ValueError:
                    price = 0
                category = row.get('category_code') or row.get('category', '')
                stock = random.randint(0, 1000)
                warranty = random.randint(6, 24)
                writers['products'].writerow([prod_id, brand, price, category, stock, warranty])
                seen_products.add(prod_id)

            # Orders (skip views)
            if event_type != 'view':
                event_time = row.get('event_time', '')
                try:
                    dt = datetime.strptime(event_time, '%Y-%m-%d %H:%M:%S %Z')
                except ValueError:
                    continue
                writers['orders'].writerow([user_id, prod_id, dt.strftime('%Y-%m-%d %H:%M:%S'), event_type])

            if row_count % report_interval == 0:
                print(f"Processed {row_count} rows: {len(seen_brands)} brands, {len(seen_users)} users, {len(seen_products)} products")

    # Add synthetic users to reach target
    current_user_count = len(seen_users)
    if current_user_count < TARGET_USER_COUNT:
        missing = TARGET_USER_COUNT - current_user_count
        max_id = max(seen_users) if seen_users else 0
        for idx in range(missing):
            new_id = max_id + idx + 1
            email, full_name = user_infos[idx % len(user_infos)]
            balance = round(random.random() * 10000, 2)
            writers['users'].writerow([new_id, email, balance, full_name])

    for f in files.values():
        f.close()
    return {k: v.name for k, v in files.items()}

# New: batch insert users with progress from CSV
def load_users_with_progress(conn, csv_path, batch_size=100000):
    print(f"Loading users in batches (batch size={batch_size})...")
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        total = sum(1 for _ in f) - 1
        f.seek(0)
        next(reader)
        inserted = 0
        batch = []
        cur = conn.cursor()
        for row in reader:
            batch.append((int(row[0]), row[1], float(row[2]), row[3]))
            if len(batch) >= batch_size:
                execute_values(
                    cur,
                    "INSERT INTO users(user_id, email, balance, full_name) VALUES %s ON CONFLICT DO NOTHING",
                    batch
                )
                conn.commit()
                inserted += len(batch)
                print(f"Inserted {inserted}/{total} users")
                batch.clear()
        if batch:
            execute_values(
                cur,
                "INSERT INTO users(user_id, email, balance, full_name) VALUES %s ON CONFLICT DO NOTHING",
                batch
            )
            conn.commit()
            inserted += len(batch)
            print(f"Inserted {inserted}/{total} users (final batch)")
        cur.close()

# New: batch insert orders with progress from CSV
def load_orders_with_progress(conn, csv_path, batch_size=200000):
    print(f"Loading orders in batches (batch size={batch_size})...")
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        total = sum(1 for _ in f) - 1
        f.seek(0)
        next(reader)
        inserted = 0
        batch = []
        cur = conn.cursor()
        for row in reader:
            # row: user_id, product_id, order_date, status
            user_id, prod_id, order_date, status = row
            batch.append((int(user_id), int(prod_id), order_date, status))
            if len(batch) >= batch_size:
                execute_values(
                    cur,
                    "INSERT INTO orders(user_id, product_id, order_date, status) VALUES %s",
                    batch
                )
                conn.commit()
                inserted += len(batch)
                print(f"Inserted {inserted}/{total} orders")
                batch.clear()
        if batch:
            execute_values(
                cur,
                "INSERT INTO orders(user_id, product_id, order_date, status) VALUES %s",
                batch
            )
            conn.commit()
            inserted += len(batch)
            print(f"Inserted {inserted}/{total} orders (final batch)")
        cur.close()

# Bulk load via COPY and finalize
def load_into_db(conn, csv_paths):
    print("Copying staging data into PostgreSQL...")
    with conn.cursor() as cur:
        setup_staging(cur)

        print("-> Copying manufacturers...")
        cur.copy_expert(
            "COPY staging_manufacturers(name) FROM STDIN WITH CSV HEADER",
            open(csv_paths['manufacturers'], 'r', encoding='utf-8')
        )

        print("-> Copying products...")
        cur.copy_expert(
            "COPY staging_products(product_id, manufacturer_name, price, category, stock_quantity, warranty_period)" \
            " FROM STDIN WITH CSV HEADER",
            open(csv_paths['products'], 'r', encoding='utf-8')
        )

        print("-> Copying staging orders...")
        cur.copy_expert(
            "COPY staging_orders(user_id, product_id, order_date, status) FROM STDIN WITH CSV HEADER",
            open(csv_paths['orders'], 'r', encoding='utf-8')
        )

        print("-> Inserting manufacturers...")
        cur.execute("""
            INSERT INTO manufacturers(name)
            SELECT DISTINCT name FROM staging_manufacturers
            ON CONFLICT (name) DO NOTHING;
        """
        )

        print("-> Inserting products...")
        cur.execute("""
            INSERT INTO products(product_id, manufacturer_id, price, category, stock_quantity, warranty_period)
            SELECT sp.product_id, m.manufacturer_id, sp.price, sp.category, sp.stock_quantity, sp.warranty_period
            FROM staging_products sp
            JOIN manufacturers m ON m.name = sp.manufacturer_name
            ON CONFLICT (product_id) DO NOTHING;
        """
        )

        print("-> Inserting users in batches with progress...")
    load_users_with_progress(conn, csv_paths['users'])

    # Batch insert orders
    print("-> Inserting orders in batches with progress...")
    load_orders_with_progress(conn, csv_paths['orders'])

    conn.commit()
    print("✅ Data successfully copied into PostgreSQL.")

# Main flow
def main():
    config = load_config()
    conn = init_db(config)
    try:
        user_infos = load_user_infos('../dataset/user_emails_full_names.csv')
        csv_paths = generate_staging_files('../dataset/big_dataset1.csv', user_infos)
        load_into_db(conn, csv_paths)
        print("Data import completed successfully.")
    except Exception as e:
        conn.rollback()
        print(f"Error during import: {e}")
    finally:
        conn.close()

if __name__ == '__main__':
    main()
