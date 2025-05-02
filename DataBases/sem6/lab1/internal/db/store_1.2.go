package db

import (
	"database/sql"
	"fmt"
	"sort"

	_ "github.com/lib/pq"
)

func PlaceOrder(tx *sql.Tx, userID, productID, qty int) error {
	// 1. Получаем цену и остаток
	var price float64
	var stock int
	err := tx.QueryRow(
		`SELECT price, stock_quantity
           FROM products 
          WHERE product_id = $1 
            FOR UPDATE`, productID,
	).Scan(&price, &stock)
	if err != nil {
		return err
	}
	if stock < qty {
		return fmt.Errorf("not enough stock")
	}

	// 2. Проверяем баланс
	var balance float64
	err = tx.QueryRow(
		`SELECT balance 
           FROM users 
          WHERE user_id = $1 
            FOR UPDATE`, userID,
	).Scan(&balance)
	if err != nil {
		return err
	}
	total := price * float64(qty)
	if balance < total {
		return fmt.Errorf("not enough funds")
	}

	// 3. Списываем у пользователя
	if _, err = tx.Exec(
		`UPDATE users 
            SET balance = balance - $1 
          WHERE user_id = $2`, total, userID,
	); err != nil {
		return err
	}

	// 4. Урезаем склад
	if _, err = tx.Exec(
		`UPDATE products 
            SET stock_quantity = stock_quantity - $1 
          WHERE product_id = $2`, qty, productID,
	); err != nil {
		return err
	}

	// 5. Вставляем заказ
	_, err = tx.Exec(
		`INSERT INTO orders(user_id, product_id, order_date, status) 
             VALUES($1, $2, NOW(), 'purchase')`,
		userID, productID,
	)
	return err
}

func TransferFunds(tx *sql.Tx, fromID, toID int, amount float64) error {
	// блокируем сначала меньший ID
	ids := []int{fromID, toID}
	sort.Ints(ids)
	for _, uid := range ids {
		if _, err := tx.Exec(
			`SELECT 1 FROM users WHERE user_id = $1 FOR UPDATE`, uid,
		); err != nil {
			return err
		}
	}
	// списываем и добавляем
	if _, err := tx.Exec(
		`UPDATE users SET balance = balance - $1 WHERE user_id = $2`,
		amount, fromID,
	); err != nil {
		return err
	}
	if _, err := tx.Exec(
		`UPDATE users SET balance = balance + $1 WHERE user_id = $2`,
		amount, toID,
	); err != nil {
		return err
	}
	return nil
}

func RestockProduct(tx *sql.Tx, productID int, qty int) error {
	_, err := tx.Exec(`
		UPDATE products
		SET stock_quantity = stock_quantity + $1
		WHERE product_id = $2
	`, qty, productID)
	return err
}

func (s *Store) PerformTransaction(isolationLevel string) error {
	// Начинаем транзакцию
	tx, err := s.DB.Begin()
	if err != nil {
		return err
	}

	// Устанавливаем уровень изоляции внутри транзакции
	_, err = tx.Exec(fmt.Sprintf("SET TRANSACTION ISOLATION LEVEL %s", isolationLevel))
	if err != nil {
		tx.Rollback()
		return err
	}

	// Пример 1: Покупка товара (PlaceOrder)
	err = PlaceOrder(tx, 530496790 /*user*/, 5000088 /*product*/, 2 /*qty*/)
	if err != nil {
		tx.Rollback()
		return err
	}

	// Пример 2: Пополнение товара (RestockProduct)
	err = RestockProduct(tx, 5000088 /*product*/, 10 /*добавить*/)
	if err != nil {
		tx.Rollback()
		return err
	}

	// Подтверждение транзакции
	if err := tx.Commit(); err != nil {
		return err
	}

	// Выводим результат: баланс пользователя и остаток товара
	var balance float64
	err = s.DB.QueryRow("SELECT balance FROM users WHERE user_id = $1", 530496790).Scan(&balance)
	if err != nil {
		return err
	}

	var quantity int
	err = s.DB.QueryRow("SELECT stock_quantity FROM products WHERE product_id = $1", 5000088).Scan(&quantity)
	if err != nil {
		return err
	}

	fmt.Printf("User 530496790 balance: %.2f\n", balance)
	fmt.Printf("Product 5000088 quantity: %d\n", quantity)

	return nil
}

func (s *Store) PerformTransfer(isolationLevel string, fromID, toID int, amount float64) error {
	// Начинаем транзакцию
	tx, err := s.DB.Begin()
	if err != nil {
		return err
	}

	// Устанавливаем уровень изоляции внутри транзакции
	_, err = tx.Exec(fmt.Sprintf("SET TRANSACTION ISOLATION LEVEL %s", isolationLevel))
	if err != nil {
		tx.Rollback()
		return err
	}

	// Перевод средств
	err = TransferFunds(tx, fromID, toID, amount)
	if err != nil {
		tx.Rollback()
		return err
	}

	// Подтверждение транзакции
	if err := tx.Commit(); err != nil {
		return err
	}

	// Показываем балансы после перевода
	var fromBalance, toBalance float64
	err = s.DB.QueryRow("SELECT balance FROM users WHERE user_id = $1", fromID).Scan(&fromBalance)
	if err != nil {
		return err
	}
	err = s.DB.QueryRow("SELECT balance FROM users WHERE user_id = $1", toID).Scan(&toBalance)
	if err != nil {
		return err
	}

	fmt.Printf("User %d balance: %.2f\n", fromID, fromBalance)
	fmt.Printf("User %d balance: %.2f\n", toID, toBalance)

	return nil
}
