package main

import (
	"db_lab1/internal/db"
	"fmt"
	"log"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)

	store, err := db.NewStore()
	if err != nil {
		log.Fatalf("Ошибка подключения к БД: %v", err)
	}
	defer store.Close()

	fmt.Println("=== SERIALIZABLE ===")
	if err := store.PerformTransaction("SERIALIZABLE"); err != nil {
		log.Fatalf("Ошибка транзакции: %v", err)
	}

	fmt.Println("=== REPEATABLE READ ===")
	err = store.PerformTransfer("REPEATABLE READ", 520088904, 561587266, 10.0)
	if err != nil {
		log.Printf("Ошибка перевода: %v", err)
	}

	fmt.Println("\n=== Non‑Repeatable Read Demo ===")
	store.DemoNonRepeatableRead()

	fmt.Println("\n=== Phantom Read Demo ===")
	store.DemoPhantomRead()

}
