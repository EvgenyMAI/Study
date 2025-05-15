import csv
import os
import uuid
import random
from faker import Faker
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

# Конфигурация
CONFIG = {
    "output_file": "./dataset/generated_users.csv",
    "total_records": 5_000_000,
    "batch_size": 50_000,
    "email_domains": ['tech.org', 'mail.io', 'cloud.net', 'data.ai', 'web.digital'],
    "max_workers": 6
}

class UserDataGenerator:
    def __init__(self):
        self.fake = Faker(['ru_RU', 'en_US'])
        
    def generate_user(self) -> Dict[str, str]:
        """Генерирует данные одного пользователя"""
        return {
            "email": f"{uuid.uuid4().hex[:12]}@{random.choice(CONFIG['email_domains'])}",
            "full_name": self.fake.unique.name()
        }

def file_exists(filepath: str) -> bool:
    """Проверяет существование файла"""
    return os.path.isfile(filepath)

def count_lines(filepath: str) -> int:
    """Считает количество строк в файле"""
    if not file_exists(filepath):
        return 0
    with open(filepath, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f) - 1  # Исключаем заголовок

def generate_users_batch(batch_size: int) -> List[Dict[str, str]]:
    """Генерирует батч пользователей"""
    generator = UserDataGenerator()
    return [generator.generate_user() for _ in range(batch_size)]

def write_to_csv(filepath: str, data: List[Dict[str, str]], mode: str = 'a'):
    """Записывает данные в CSV"""
    with open(filepath, mode, newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['email', 'full_name'])
        if mode == 'w':
            writer.writeheader()
        writer.writerows(data)

def main():
    existing_lines = count_lines(CONFIG["output_file"])
    remaining = CONFIG["total_records"] - existing_lines
    
    if remaining <= 0:
        print(f"Данные уже сгенерированы: {existing_lines} записей")
        return

    print(f"Генерация {remaining} пользователей...")

    # Инициализация файла
    if existing_lines == 0:
        write_to_csv(CONFIG["output_file"], [], mode='w')

    # Параллельная генерация
    with ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
        futures = []
        for _ in range(0, remaining, CONFIG["batch_size"]):
            current_batch = min(CONFIG["batch_size"], remaining)
            futures.append(executor.submit(generate_users_batch, current_batch))
            remaining -= current_batch

        for i, future in enumerate(as_completed(futures), 1):
            batch = future.result()
            write_to_csv(CONFIG["output_file"], batch)
            print(f"Записано {len(batch)} записей (батч {i}/{len(futures)})")

    print(f"Генерация завершена. Итого: {CONFIG['total_records']} записей")

if __name__ == '__main__':
    main()