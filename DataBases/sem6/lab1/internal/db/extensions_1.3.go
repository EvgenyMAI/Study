package db

import (
	"context"
	"fmt"
	"log"
	"time"

	_ "github.com/lib/pq"
)

// EnsureFuzzyExtensionsAndIndexes создаёт расширения и GIN‑индексы, если их нет
func (s *Store) EnsureFuzzyExtensionsAndIndexes(ctx context.Context) error {
	stmts := []string{
		`CREATE EXTENSION IF NOT EXISTS pg_trgm`,
		`CREATE EXTENSION IF NOT EXISTS pg_bigm`,
		`CREATE EXTENSION IF NOT EXISTS pgcrypto`,

		`CREATE INDEX IF NOT EXISTS idx_manufacturers_name_trgm ON manufacturers USING gin (name gin_trgm_ops)`,
		`CREATE INDEX IF NOT EXISTS idx_manufacturers_name_bigm ON manufacturers USING gin (name gin_bigm_ops)`,

		`CREATE INDEX IF NOT EXISTS idx_products_category_trgm ON products USING gin (category gin_trgm_ops)`,
		`CREATE INDEX IF NOT EXISTS idx_products_category_bigm ON products USING gin (category gin_bigm_ops)`,
	}
	for i, ddl := range stmts {
		if _, err := s.DB.ExecContext(ctx, ddl); err != nil {
			return fmt.Errorf("step %d failed (%q): %w", i+1, ddl, err)
		}
	}
	return nil
}

// Manufacturer модель производителя
type Manufacturer struct {
	ID   int
	Name string
}

// Product модель товара
type Product struct {
	ID       int
	Category string
	Price    float64
}

// FuzzySearchManufacturers ищет производителей по похожему имени
func (s *Store) FuzzySearchManufacturers(
	ctx context.Context,
	term string,
	method string, // "trgm" или "bigm"
	thresh float64,
	limit int,
) ([]Manufacturer, error) {
	// Используем оператор similarity, GIN‑индекс сам подхватится
	sqlQuery := fmt.Sprintf(
		`SELECT manufacturer_id, name FROM manufacturers WHERE similarity(name, $1) > $2 ORDER BY similarity(name, $1) DESC LIMIT $3`,
	)
	rows, err := s.DB.QueryContext(ctx, sqlQuery, term, thresh, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var res []Manufacturer
	for rows.Next() {
		var m Manufacturer
		if err := rows.Scan(&m.ID, &m.Name); err != nil {
			return nil, err
		}
		res = append(res, m)
	}
	return res, nil
}

// FuzzySearchProducts ищет товары по похожей категории
func (s *Store) FuzzySearchProducts(
	ctx context.Context,
	term string,
	method string,
	thresh float64,
	limit int,
) ([]Product, error) {
	sqlQuery := fmt.Sprintf(
		`SELECT product_id, category, price FROM products WHERE similarity(category, $1) > $2 ORDER BY similarity(category, $1) DESC LIMIT $3`,
	)
	rows, err := s.DB.QueryContext(ctx, sqlQuery, term, thresh, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var res []Product
	for rows.Next() {
		var p Product
		if err := rows.Scan(&p.ID, &p.Category, &p.Price); err != nil {
			return nil, err
		}
		res = append(res, p)
	}
	return res, nil
}

// CreateUserEncrypted сохраняет пользователя, шифруя email через pgcrypto
func (s *Store) CreateUserEncrypted(
	ctx context.Context,
	userID int,
	email, name string,
	balance float64,
	key string,
) error {
	_, err := s.DB.ExecContext(ctx, `
        INSERT INTO users(user_id, email_enc, balance, name)
        VALUES ($1, pgp_sym_encrypt($2, $5), $3, $4)
    `, userID, email, balance, name, key)
	return err
}

// GetUserDecrypted возвращает расшифрованный email и остальные поля
func (s *Store) GetUserDecrypted(
	ctx context.Context,
	userID int,
	key string,
) (email, name string, balance float64, err error) {
	row := s.DB.QueryRowContext(ctx, `
        SELECT pgp_sym_decrypt(email_enc, $2)::text AS email, name, balance
        FROM users
        WHERE user_id = $1
    `, userID, key)
	err = row.Scan(&email, &name, &balance)
	return
}

// EnsureEncryptedUsersTable создаёт таблицу encrypted_users с внешним ключом на users.user_id
func (s *Store) EnsureEncryptedUsersTable(ctx context.Context) error {
	ddl := `
    CREATE TABLE IF NOT EXISTS encrypted_users (
        user_id         BIGINT PRIMARY KEY,
        encrypted_email BYTEA NOT NULL,
        CONSTRAINT fk_user_id FOREIGN KEY (user_id)
            REFERENCES users(user_id)
            ON DELETE CASCADE
    );
    `
	if _, err := s.DB.ExecContext(ctx, ddl); err != nil {
		return fmt.Errorf("create encrypted_users table: %w", err)
	}
	return nil
}

// EncryptRandomUsers берёт count случайных записей из users и сохраняет их зашифрованные email в encrypted_users
func (s *Store) EncryptRandomUsers(ctx context.Context, key string, total int) error {
	const batchSize = 10_000

	start := time.Now()

	log.Println("Создание таблицы encrypted_users (если не существует)...")
	_, err := s.DB.ExecContext(ctx, `
        CREATE TABLE IF NOT EXISTS encrypted_users (
            user_id INT PRIMARY KEY,
            email_enc BYTEA
        )
    `)
	if err != nil {
		return fmt.Errorf("create table: %w", err)
	}

	log.Printf("Начинаем шифрование %d пользователей батчами по %d...\n", total, batchSize)

	for offset := 0; offset < total; offset += batchSize {
		select {
		case <-ctx.Done():
			return fmt.Errorf("context canceled: %w", ctx.Err())
		default:
		}

		log.Printf("Шифруем пользователей: %d – %d...", offset+1, offset+batchSize)

		_, err := s.DB.ExecContext(ctx, `
            INSERT INTO encrypted_users (user_id, email_enc)
            SELECT user_id, pgp_sym_encrypt(email, $1)
            FROM users
            ORDER BY random()
            LIMIT $2
            ON CONFLICT (user_id) DO NOTHING
        `, key, batchSize)
		if err != nil {
			return fmt.Errorf("batch offset %d failed: %w", offset, err)
		}
	}

	log.Printf("✅ Шифрование завершено за %s\n", time.Since(start))
	return nil
}

// GetDecryptedEmailByID возвращает расшифрованный email по user_id без использования контекста
func (s *Store) GetDecryptedEmailByID(
	userID int,
	key string,
) (string, error) {
	var email string
	err := s.DB.QueryRow(`
        SELECT pgp_sym_decrypt(email_enc, $2)::text
        FROM encrypted_users
        WHERE user_id = $1
    `, userID, key).Scan(&email)

	if err != nil {
		return "", fmt.Errorf("decrypt email for user %d: %w", userID, err)
	}
	return email, nil
}
