package main

import (
	"db_lab1/internal/db"
	"flag"
	"log"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)

	mode := flag.String("mode", "benchmark-index", "режим: create-indexes, drop-indexes, benchmark-index, benchmark-noindex")
	flag.Parse()

	store, err := db.NewStore()
	if err != nil {
		log.Fatalf("Ошибка подключения к БД: %v", err)
	}
	defer store.Close()

	switch *mode {

	case "create-indexes":
		if err := store.CreateIndexes(); err != nil {
			log.Fatalf("Ошибка при создании индексов: %v", err)
		}

	case "drop-indexes":
		if err := store.DropIndexes(); err != nil {
			log.Fatalf("Ошибка при удалении индексов: %v", err)
		}

	case "benchmark-index":
		if err := store.Benchmark("с индексами", true); err != nil {
			log.Fatalf("Ошибка бенчмарка с индексами: %v", err)
		}

	case "benchmark-noindex":
		if err := store.Benchmark("без индексов", false); err != nil {
			log.Fatalf("Ошибка бенчмарка без индексов: %v", err)
		}

	default:
		log.Fatalf("Неизвестный режим: %s", *mode)
	}
}
