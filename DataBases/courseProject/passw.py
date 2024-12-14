import bcrypt
import psycopg2
from settings import DB_CONFIG

# Подключение к базе данных
conn = psycopg2.connect(**DB_CONFIG)

def hash_password(password):
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

# Получение всех пользователей с их паролями
query_select = "SELECT user_id, password FROM users"

# Обновление пароля
query_update = """
    UPDATE users
    SET password = %s
    WHERE user_id = %s
"""

try:
    with conn:
        with conn.cursor() as cursor:
            # Извлечение всех пользователей
            cursor.execute(query_select)
            users = cursor.fetchall()

            for user_id, password in users:
                print(f"Проверяем пользователя ID {user_id} с паролем: {password}")

                # Если пароль уже захэширован (определяется по длине), пропускаем
                if password.startswith("$2b$"):
                    print(f"Пользователь ID {user_id}: пароль уже захэширован, пропускаем.")
                    continue

                # Хэшируем и обновляем пароль
                hashed_password = hash_password(password)
                cursor.execute(query_update, (hashed_password, user_id))
                print(f"Пользователь ID {user_id}: пароль обновлен.")

            conn.commit()
            print("Все изменения зафиксированы.")
except Exception as e:
    print(f"Ошибка: {e}")
finally:
    conn.close()