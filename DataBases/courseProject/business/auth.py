import repositories.users
import bcrypt

class Authorize:
    def __init__(self):
        """
        Инициализация авторизации. Загружаем данные пользователей (email и пароль).
        """
        self.users = self.load_users()

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
        hashed_password = self.users.get(email)

        if not hashed_password:
            raise ValueError("Пользователь с таким email не найден.")

        # Проверка пароля
        return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))

    def get_user_data(self, email: str) -> dict:
        """
        Возвращает информацию о пользователе по email.
        """
        user_data = repositories.users.get_user_by_email(email)
        if not user_data:
            raise ValueError("Пользователь с таким email не найден.")
        return user_data[0]