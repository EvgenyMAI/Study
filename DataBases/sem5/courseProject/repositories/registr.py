from pandas import DataFrame
import psycopg2
from settings import DB_CONFIG
import bcrypt

def registration(user: DataFrame) -> int:
    """
    Регистрация нового пользователя в базе данных.
    """
    query = """
        INSERT INTO users (email, password, first_name, last_name, role)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING user_id
    """
    # Хэширование пароля
    hashed_password = bcrypt.hashpw(user["password"].item().encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    user["password"] = hashed_password

    params = (
        user["email"].iloc[0],
        user["password"].iloc[0],
        user["first_name"].iloc[0],
        user["last_name"].iloc[0],
        user["role"].iloc[0],
    )

    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchone()[0]