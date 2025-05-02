import csv
import os
from faker import Faker
import uuid
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Настройки ---
FILENAME = 'users.csv'
TOTAL_RECORDS = 5_000_000
BATCH_SIZE = 10_000
DOMAINS = ['gmail.com', 'yahoo.com', 'outlook.com', 'mail.net', 'yandex.ru', 'hotmail.com']
MAX_WORKERS = 8

fake = Faker()

def count_existing_records():
    """Считает количество строк (без заголовка) в бинарном режиме."""
    if not os.path.exists(FILENAME):
        return 0
    with open(FILENAME, 'rb') as f:
        total_lines = f.read().count(b'\n')
    return max(0, total_lines - 1)

def load_existing_emails():
    """
    Читает существующие email, игнорируя NUL-байты.
    Пропускаем null-символы перед разбором CSV.
    """
    emails = set()
    if not os.path.exists(FILENAME):
        return emails

    def sanitize_lines(f):
        for raw in f:
            # избавляемся от всех '\0'
            yield raw.replace('\0', '')

    with open(FILENAME, 'r', encoding='utf-8', errors='ignore', newline='') as f:
        reader = csv.DictReader(sanitize_lines(f))
        for row in reader:
            email = row.get('email')
            if email:
                emails.add(email)
    return emails

def make_batch(n_records):
    users = []
    for _ in range(n_records):
        full_name = fake.name()
        email = f"{uuid.uuid4().hex}@{random.choice(DOMAINS)}"
        users.append({'email': email, 'full_name': full_name})
    return users

def write_batch(users):
    with open(FILENAME, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['email', 'full_name'])
        writer.writerows(users)

def main():
    already = count_existing_records()
    to_go = TOTAL_RECORDS - already
    if to_go <= 0:
        print(f"✔ В файле уже {already} записей, больше ничего не нужно.")
        return

    print(f"Найдено {already} записей, генерируем ещё {to_go}…")

    # Если пустой файл — создаём и пишем заголовок
    if already == 0:
        with open(FILENAME, 'w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=['email', 'full_name']).writeheader()

    # Загружаем существующие email (с очисткой NUL)
    existing_emails = load_existing_emails()

    # Готовим списки размеров батчей
    full_batches = to_go // BATCH_SIZE
    last_batch = to_go % BATCH_SIZE
    batch_sizes = [BATCH_SIZE]*full_batches + ([last_batch] if last_batch else [])

    total_batches = len(batch_sizes)
    print(f"Будет записано {total_batches} пакетов по {BATCH_SIZE} (последний {last_batch or BATCH_SIZE}).")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futures = {
            exe.submit(make_batch, size): idx
            for idx, size in enumerate(batch_sizes, start=1)
        }
        for future in as_completed(futures):
            idx = futures[future]
            users = future.result()

            # Подстраховка на редкие совпадения (хоть с uuid они маловероятны)
            for u in users:
                if u['email'] in existing_emails:
                    u['email'] = f"{uuid.uuid4().hex}@{random.choice(DOMAINS)}"
                existing_emails.add(u['email'])

            write_batch(users)
            print(f"✅ Пакет {idx}/{total_batches} ({len(users)} записей) записан.")

    print(f"🎉 В итоге в {FILENAME} — ровно {TOTAL_RECORDS} записей.")

if __name__ == '__main__':
    main()