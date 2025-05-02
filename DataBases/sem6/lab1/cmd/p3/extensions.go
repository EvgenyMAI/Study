package main

import (
	"context"
	"db_lab1/internal/db"
	"fmt"
	"log"
	"time"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)

	store, err := db.NewStore()
	if err != nil {
		log.Fatalf("Ошибка подключения к БД: %v", err)
	}
	defer store.Close()

	// Общий контекст с таймаутом для поисковых и крипто-операций
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// --- Новые блоки:
	fmt.Println("\n=== Fuzzy Search Manufacturers ===")
	ms, err := store.FuzzySearchManufacturers(ctx, "maliz", "trgm", 0.2, 10)
	if err != nil {
		log.Fatalf("Fuzzy Manufacturers: %v", err)
	}
	for _, m := range ms {
		fmt.Printf("ID=%d Name=%s\n", m.ID, m.Name)
	}

	fmt.Println("\n=== Fuzzy Search Products ===")
	ps, err := store.FuzzySearchProducts(ctx, "smartphone", "bigm", 0.3, 5)
	if err != nil {
		log.Fatalf("Fuzzy Products: %v", err)
	}
	for _, p := range ps {
		fmt.Printf("ID=%d Category=%s Price=%.2f\n", p.ID, p.Category, p.Price)
	}

	new_ctx, cancel := context.WithTimeout(context.Background(), 500*time.Second)
	defer cancel()
	if err := store.EncryptRandomUsers(new_ctx, "my-secret-key", 10000); err != nil {
		log.Fatalf("Ошибка шифрования рандомных пользователей: %v", err)
	}

	email, err := store.GetDecryptedEmailByID(581234831, "my-secret-key")
	if err != nil {
		log.Printf("Ошибка расшифровки: %v", err)
	} else {
		fmt.Printf("Email пользователя 581234831: %s\n", email)
	}
}
