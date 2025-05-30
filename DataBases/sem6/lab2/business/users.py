import pandas as pd
import repositories.users

def get_users() -> pd.DataFrame:
    """
    Получить список всех пользователей в виде DataFrame.
    """
    users = repositories.users.get_all_users()
    return pd.DataFrame(users)