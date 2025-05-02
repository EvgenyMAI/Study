import pandas as pd
import repositories.registr

class Registration:
    def registr(self, user: pd.DataFrame) -> int:
        """
        Обрабатывает данные для регистрации пользователя.
        """
        # Проверка обязательных полей
        required_fields = ["email", "password", "first_name", "last_name", "role"]
        for field in required_fields:
            if field not in user.columns or user[field].iloc[0] is None:
                raise ValueError(f"Поле '{field}' является обязательным для регистрации.")

        # Передача данных в репозиторий
        return repositories.registr.registration(user)