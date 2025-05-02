import pandas as pd
import repositories.users

def get_user(email: str) -> pd.DataFrame:
    """
    Получить данные пользователя в виде DataFrame.
    """
    user = repositories.users.get_user_by_email(email)
    if not user:
        return None
    return pd.DataFrame([user])