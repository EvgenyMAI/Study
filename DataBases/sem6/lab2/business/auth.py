import repositories.users
import bcrypt
import jwt
import time
from repositories.redis_client import redis_client
import os
from dotenv import load_dotenv

load_dotenv("../environment/env.env")

class Authorize:
    def __init__(self):
        """
        Инициализация авторизации. Загружаем данные пользователей (email и пароль).
        """
        self.redis = redis_client.get_connection()
        self.token_ttl = int(os.getenv("TOKEN_TTL"))
        self.session_ttl = int(os.getenv("SESSION_TTL"))
        self.secret_key = os.getenv("SECRET_KEY", "qwery228")

    def load_users(self) -> dict:
        """
        Загружает пользователей из базы данных и возвращает их в виде словаря.
        Ключ - email, значение - хэш пароля.
        """
        users = repositories.users.get_all_users()
        return {user["email"]: user["password"] for user in users}

    def auth(self, email: str, password: str) -> bool:
        """
        Проверяет введенные данные для авторизации.
        """
        # Пытаемся получить хэш пароля из Redis
        redis_password = self.redis.hget(f"user:{email}", "password")
        
        if redis_password:
            # Проверяем пароль с хэшем из Redis
            try:
                return bcrypt.checkpw(password.encode("utf-8"), redis_password.encode("utf-8"))
            except:
                # Если что-то пошло не так с Redis-данными, пробуем через базу
                pass

        # Если нет в Redis или ошибка, загружаем из базы
        user_data = repositories.users.get_user_by_email(email)
        if not user_data:
            raise ValueError("Пользователь с таким email не найден.")

        # Получаем хэш пароля из базы
        db_password = repositories.users.get_user_password_by_email(email)
        if not db_password:
            raise ValueError("Ошибка при получении данных пользователя.")

        # Проверяем пароль
        password_matches = bcrypt.checkpw(password.encode("utf-8"), db_password.encode("utf-8"))
        
        if password_matches:
            # Сохраняем данные пользователя в Redis
            self.redis.hset(f"user:{email}", mapping={
                "password": db_password,
                "first_name": user_data["first_name"],
                "last_name": user_data["last_name"],
                "role": user_data["role"],
                "user_id": str(user_data["user_id"])
            })
            self.redis.expire(f"user:{email}", self.session_ttl)
        
        return password_matches

    def generate_token(self, email: str) -> str:
        """
        Генерирует JWT токен для пользователя и сохраняет его в Redis.
        """
        payload = {
            "email": email,
            "exp": time.time() + self.token_ttl
        }
        token = jwt.encode(payload, self.secret_key, algorithm="HS256")
        
        # Сохраняем токен в Redis с TTL
        self.redis.set(f"token:{token}", email, ex=self.token_ttl)
        return token

    def validate_token(self, token: str) -> bool:
        """
        Проверяет валидность токена.
        """
        # Проверяем наличие токена в Redis
        return self.redis.exists(f"token:{token}") == 1

    def get_user_data(self, email: str) -> dict:
        """
        Возвращает информацию о пользователе по email, сначала проверяя Redis.
        """
        # Пытаемся получить из Redis
        user_data = self.redis.hgetall(f"user:{email}")
        
        if not user_data:
            # Если нет в Redis, загружаем из базы
            user_data = repositories.users.get_user_by_email(email)
            if not user_data:
                raise ValueError("Пользователь с таким email не найден.")
            
            # Сохраняем в Redis
            self.redis.hset(f"user:{email}", mapping=user_data)
            self.redis.expire(f"user:{email}", self.session_ttl)
        else:
            # Преобразуем данные из Redis в тот же формат, что и из базы
            user_data = {
                "user_id": int(user_data.get("user_id", 0)),
                "email": email,
                "first_name": user_data.get("first_name"),
                "last_name": user_data.get("last_name"),
                "role": user_data.get("role"),
                "password": user_data.get("password")
            }
        
        return user_data

    def logout(self, token: str) -> None:
        """
        Удаляет токен из Redis при выходе из системы.
        """
        self.redis.delete(f"token:{token}")