import psycopg2
from settings import DB_CONFIG
import pandas as pd

def get_user_favorites(user_id):
    """
    Получает список избранных статей для конкретного пользователя.

    :param user_id: ID пользователя
    :return: Список избранных статей
    """
    query = """
        SELECT a.article_id, a.title, 
               array_agg(DISTINCT CONCAT(au.first_name, ' ', au.last_name)) AS authors, 
               a.publication_year, a.link,
               array_agg(DISTINCT k.keyword) AS keywords  -- Добавляем агрегацию для ключевых слов
        FROM articles a
        LEFT JOIN article_authors aa ON a.article_id = aa.article_id
        LEFT JOIN authors au ON aa.author_id = au.author_id
        LEFT JOIN article_keywords ak ON a.article_id = ak.article_id
        LEFT JOIN keywords k ON ak.keyword_id = k.keyword_id
        LEFT JOIN user_favorites uf ON a.article_id = uf.article_id
        WHERE uf.user_id = %s
        GROUP BY a.article_id
        ORDER BY a.publication_year DESC;
    """
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (user_id,))
            # Преобразуем результат в DataFrame
            articles = cur.fetchall()
            columns = ['article_id', 'title', 'authors', 'publication_year', 'link', 'keywords']
            return pd.DataFrame(articles, columns=columns)

def add_to_favorites(user_id, article_id):
    """
    Добавляет статью в список избранных для пользователя, если её там нет.

    :param user_id: ID пользователя
    :param article_id: ID статьи
    """
    if is_article_in_favorites(user_id, article_id):
        raise ValueError("Эта статья уже добавлена в избранное.")

    query = """
        INSERT INTO user_favorites (user_id, article_id)
        VALUES (%s, %s);
    """
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (user_id, article_id))
        conn.commit()

def remove_from_favorites(user_id, article_id):
    """
    Удаляет статью из списка избранных для пользователя.

    :param user_id: ID пользователя
    :param article_id: ID статьи
    """
    query = """
        DELETE FROM user_favorites 
        WHERE user_id = %s AND article_id = %s;
    """
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (user_id, article_id))
        conn.commit()

def is_article_in_favorites(user_id, article_id):
    """
    Проверяет, добавлена ли статья в избранное у пользователя.

    :param user_id: ID пользователя
    :param article_id: ID статьи
    :return: True, если статья в избранном, иначе False
    """
    query = """
        SELECT 1
        FROM user_favorites
        WHERE user_id = %s AND article_id = %s;
    """
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (user_id, article_id))
            result = cur.fetchone()
            return result is not None

def get_article_id_by_title(title):
    """
    Получает article_id по названию статьи.

    :param title: Название статьи
    :return: ID статьи
    """
    query = """
        SELECT article_id
        FROM articles
        WHERE title = %s;
    """
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (title,))
            result = cur.fetchone()
            if result:
                return result[0]
            return None
