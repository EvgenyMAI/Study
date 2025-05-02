package db

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"sync"
	"time"

	_ "github.com/lib/pq"
)

// DemoNonRepeatableRead показывает, как при уровне READ COMMITTED один и тот же SELECT
// в одной транзакции может вернуть разный результат (non‑repeatable read).
func (s *Store) DemoNonRepeatableRead() {
	fmt.Println(">>> Non‑Repeatable Read (READ COMMITTED) <<<")
	var wg sync.WaitGroup
	wg.Add(2)

	// Tx1
	go func() {
		defer wg.Done()
		tx1, err := s.DB.BeginTx(context.Background(), &sql.TxOptions{Isolation: sql.LevelReadCommitted})
		if err != nil {
			log.Println("Tx1 begin:", err)
			return
		}
		defer tx1.Rollback()

		var price1, price2 float64
		// Первый SELECT
		if err := tx1.QueryRow("SELECT price FROM products WHERE product_id = $1", 1003461).Scan(&price1); err != nil {
			log.Println("Tx1 first read:", err)
			return
		}
		fmt.Printf("Tx1 first read price = %.2f\n", price1)

		time.Sleep(500 * time.Millisecond) // ждём, пока Tx2 обновит

		// Второй SELECT
		if err := tx1.QueryRow("SELECT price FROM products WHERE product_id = $1", 1003461).Scan(&price2); err != nil {
			log.Println("Tx1 second read:", err)
			return
		}
		fmt.Printf("Tx1 second read price = %.2f\n", price2)

		tx1.Commit()
	}()

	// Tx2
	go func() {
		defer wg.Done()
		time.Sleep(100 * time.Millisecond) // чтобы Tx1 сделал первый SELECT

		tx2, err := s.DB.BeginTx(context.Background(), &sql.TxOptions{Isolation: sql.LevelReadCommitted})
		if err != nil {
			log.Println("Tx2 begin:", err)
			return
		}
		defer tx2.Rollback()

		if _, err := tx2.Exec("UPDATE products SET price = price * 1.10 WHERE product_id = $1", 1003461); err != nil {
			log.Println("Tx2 update:", err)
			return
		}
		tx2.Commit()
		fmt.Println("Tx2 committed price update")
	}()

	wg.Wait()
}

// DemoPhantomRead показывает фантомные строки: COUNT(*) меняется внутри одной транзакции.
func (s *Store) DemoPhantomRead() {
	fmt.Println(">>> Phantom Read (READ COMMITTED) <<<")
	var wg sync.WaitGroup
	wg.Add(2)

	// Tx1
	go func() {
		defer wg.Done()
		tx1, err := s.DB.BeginTx(context.Background(), &sql.TxOptions{Isolation: sql.LevelReadCommitted})
		if err != nil {
			log.Println("Tx1 begin:", err)
			return
		}
		defer tx1.Rollback()

		var cnt1, cnt2 int
		if err := tx1.QueryRow("SELECT COUNT(*) FROM orders WHERE product_id = $1", 1003461).Scan(&cnt1); err != nil {
			log.Println("Tx1 first count:", err)
			return
		}
		fmt.Printf("Tx1 first count = %d\n", cnt1)

		time.Sleep(500 * time.Millisecond) // ждём, пока Tx2 вставит новую строку

		if err := tx1.QueryRow("SELECT COUNT(*) FROM orders WHERE product_id = $1", 1003461).Scan(&cnt2); err != nil {
			log.Println("Tx1 second count:", err)
			return
		}
		fmt.Printf("Tx1 second count = %d\n", cnt2)

		tx1.Commit()
	}()

	// Tx2
	go func() {
		defer wg.Done()
		time.Sleep(100 * time.Millisecond) // даём Tx1 выполнить первый COUNT

		tx2, err := s.DB.BeginTx(context.Background(), &sql.TxOptions{Isolation: sql.LevelReadCommitted})
		if err != nil {
			log.Println("Tx2 begin:", err)
			return
		}
		defer tx2.Rollback()

		// здесь ставим свой user_id
		if _, err := tx2.Exec(
			`INSERT INTO orders(user_id, product_id, order_date, status)
             VALUES($1, $2, NOW(), 'purchase')`,
			530496790, // ваш user_id
			1003461,   // product_id
		); err != nil {
			log.Println("Tx2 insert:", err)
			return
		}
		tx2.Commit()
		fmt.Println("Tx2 committed new order")
	}()

	wg.Wait()
}
