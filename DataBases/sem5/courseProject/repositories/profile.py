import psycopg2
from settings import DB_CONFIG

def delete_user_account(user_id: int) -> bool:
    """
    Удаляет пользователя по указанному ID из базы данных, обновляя связи с его статьями,
    но оставляя статьи в базе данных.
    """
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                # Обновляем поле uploaded_by для всех статей, загруженных этим пользователем
                cur.execute("""
                    UPDATE articles
                    SET uploaded_by = NULL
                    WHERE uploaded_by = %s
                """, (user_id,))

                # Удаляем все избранные статьи пользователя
                cur.execute("DELETE FROM user_favorites WHERE user_id = %s", (user_id,))

                # Удаляем пользователя
                cur.execute("DELETE FROM users WHERE user_id = %s", (user_id,))

                # Проверка успешного удаления
                if cur.rowcount > 0:  # Если хотя бы одна запись была удалена
                    return True
                else:
                    return False  # Если пользователь не найден
    except Exception as e:
        print(f"Ошибка при удалении пользователя: {e}")
        return False