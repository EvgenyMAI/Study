package db

import (
	"database/sql"
	"log"
	"strings"
	"time"
)

// Store wraps the sql.DB and provides methods for benchmarking.
type Store struct {
	DB *sql.DB
}

// NewStore creates a Store by initializing the DB connection.
func NewStore() (*Store, error) {
	dbConn, err := InitDB()
	if err != nil {
		return nil, err
	}
	return &Store{DB: dbConn}, nil
}

// Close closes the underlying DB connection.
func (s *Store) Close() error {
	return s.DB.Close()
}

// CreateIndexes creates all needed indexes (btree, brin, gin).
func (s *Store) CreateIndexes() error {
	log.Println("Создание индексов...")

	// Настройки для ускоренного создания
	s.DB.Exec("SET maintenance_work_mem = '1GB'")
	s.DB.Exec("SET max_parallel_maintenance_workers = 4")

	indexes := []string{
		// BTree
		"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_products_price ON products(price);",

		// BRIN
		"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_brin_date ON orders USING BRIN(order_date);",

		// GIN
		"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_fullname_gin ON users USING GIN(to_tsvector('english', full_name));",
	}

	for _, stmt := range indexes {
		if _, err := s.DB.Exec(stmt); err != nil {
			log.Printf("Ошибка при создании индекса: %v", err)
			return err
		}
	}
	log.Println("Индексы успешно созданы.")
	return nil
}

// DropIndexes — на случай, если захочешь удалить всё
func (s *Store) DropIndexes() error {
	log.Println("Удаление индексов...")

	queries := []string{
		"DROP INDEX IF EXISTS idx_products_price;",
		"DROP INDEX IF EXISTS idx_orders_brin_date;",
		"DROP INDEX IF EXISTS idx_users_fullname_gin;",
	}

	for _, q := range queries {
		if _, err := s.DB.Exec(q); err != nil {
			log.Printf("Ошибка при удалении: %v", err)
			return err
		}
	}
	log.Println("Индексы удалены.")
	return nil
}

// Benchmark runs test queries and logs the execution time.
// If useIndex == false, Postgres is told not to use indexes.
func (s *Store) Benchmark(label string, useIndex bool) error {
	tests := []struct {
		name  string
		query string
	}{
		{"Price Range", "SELECT COUNT(*) FROM products WHERE price BETWEEN 500 AND 3000"},
		{"Date Range", "SELECT COUNT(*) FROM orders WHERE order_date BETWEEN '2019-11-01' AND '2019-11-02'"},
		{
			"FullName GIN Search",
			`SELECT COUNT(*) FROM users 
			 WHERE full_name IS NOT NULL 
			   AND length(full_name) < 1000 
			   AND to_tsvector('english', full_name) @@ plainto_tsquery('english', 'John')`,
		},
	}

	if !useIndex {
		s.DB.Exec("SET enable_indexscan = OFF;")
		s.DB.Exec("SET enable_bitmapscan = OFF;")
		s.DB.Exec("SET enable_seqscan = ON;")
		log.Println("⚠️ Индексы временно отключены для теста")
	} else {
		s.DB.Exec("RESET enable_indexscan;")
		s.DB.Exec("RESET enable_bitmapscan;")
		s.DB.Exec("RESET enable_seqscan;")
		log.Println("✅ Индексы будут использоваться")
	}

	log.Printf("=== Benchmark: %s ===", label)

	for _, t := range tests {
		for attempts := 0; attempts < 3; attempts++ {
			start := time.Now()
			var count int
			err := s.DB.QueryRow(t.query).Scan(&count)
			if err != nil {
				if strings.Contains(err.Error(), "recovery mode") || strings.Contains(err.Error(), "server closed") {
					log.Printf("Postgres не готов, попытка #%d...", attempts+1)
					time.Sleep(2 * time.Second)
					continue
				}
				log.Printf("Ошибка в %s: %v", t.name, err)
			} else {
				log.Printf("%s: %v (rows=%d)", t.name, time.Since(start), count)
			}
			break
		}
	}
	return nil
}
