-- Установка расширения
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS pg_bigm;

-- Таблица производителей
CREATE TABLE manufacturers (
    manufacturer_id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL
);

-- Таблица продуктов
CREATE TABLE products (
    product_id BIGINT PRIMARY KEY,
    manufacturer_id INT REFERENCES manufacturers(manufacturer_id) ON DELETE CASCADE,
    price INT,  
    category TEXT,
    stock_quantity INT,
    warranty_period INT
);

-- Таблица пользователей
CREATE TABLE users (
    user_id BIGINT PRIMARY KEY,
    email TEXT,
    balance FLOAT, 
    full_name TEXT
);

-- Таблица заказов
CREATE TABLE orders (
    order_id SERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    order_date TIMESTAMP NOT NULL,
    status VARCHAR(50) DEFAULT 'completed',
    product_id BIGINT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE CASCADE
);