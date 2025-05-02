import psycopg2
import psycopg2.extras
from settings import DB_CONFIG

def get_all_users() -> list[dict]:
    """
    Получить список всех пользователей.
    """
    query = "SELECT user_id, email, first_name, last_name, role, password FROM users;"
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query)
            return cur.fetchall()

def get_user_by_email(email: str) -> dict:
    """
    Получить данные пользователя по email.
    """
    query = """
        SELECT user_id, email, first_name, last_name, role, registration_date
        FROM users
        WHERE email = %s
    """
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, (email,))
            return cur.fetchone()

def get_user_password_by_email(email: str) -> str:
    """
    Получить хэш пароля пользователя по email.
    """
    query = "SELECT password FROM users WHERE email = %s"
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (email,))
            result = cur.fetchone()
            return result[0] if result else None