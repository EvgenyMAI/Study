import psycopg2
from settings import DB_CONFIG
import pandas as pd
from repositories.redis_client import redis_client
from business.notifications import NotificationService

def get_user_favorites(user_id):
    """
    Получает список избранных статей для пользователя с кэшированием в Redis.
    """
    redis = redis_client.get_connection()
    cache_key = f"favorites:{user_id}"
    
    # Пытаемся получить из кэша
    cached_favorites = redis.get(cache_key)
    if cached_favorites:
        return pd.read_json(cached_favorites)
    
    query = """
        SELECT a.article_id, a.title, 
               array_agg(DISTINCT CONCAT(au.first_name, ' ', au.last_name)) AS authors, 
               a.publication_year, a.link,
               array_agg(DISTINCT k.keyword) AS keywords
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
            articles = cur.fetchall()
            columns = ['article_id', 'title', 'authors', 'publication_year', 'link', 'keywords']
            df = pd.DataFrame(articles, columns=columns)
            
            # Сохраняем в кэш на 30 минут
            redis.setex(cache_key, 1800, df.to_json())
            return df

def add_to_favorites(user_id, article_id):
    """
    Добавляет статью в избранное с уведомлением и инвалидацией кэша.
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
    
    # Инвалидируем кэш избранного
    redis = redis_client.get_connection()
    redis.delete(f"favorites:{user_id}")
    redis.delete(f"favorite:{user_id}:{article_id}")
    
    # Отправляем уведомление
    article_title = get_article_title(article_id)
    notifications = NotificationService()
    notifications.send_notification(
        user_id,
        f"Статья '{article_title}' добавлена в избранное",
        "favorite"
    )

def remove_from_favorites(user_id, article_id):
    """
    Удаляет статью из избранного с уведомлением и инвалидацией кэша.
    """
    query = """
        DELETE FROM user_favorites 
        WHERE user_id = %s AND article_id = %s;
    """
    
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (user_id, article_id))
        conn.commit()
    
    # Инвалидируем кэш избранного
    redis = redis_client.get_connection()
    redis.delete(f"favorites:{user_id}")
    redis.delete(f"favorite:{user_id}:{article_id}")
    
    # Отправляем уведомление
    article_title = get_article_title(article_id)
    notifications = NotificationService()
    notifications.send_notification(
        user_id,
        f"Статья '{article_title}' удалена из избранного",
        "favorite"
    )

def is_article_in_favorites(user_id, article_id):
    """
    Проверяет наличие статьи в избранном с кэшированием в Redis.
    """
    redis = redis_client.get_connection()
    cache_key = f"favorite:{user_id}:{article_id}"
    
    # Пытаемся получить из кэша
    cached = redis.get(cache_key)
    if cached is not None:
        return cached == "1"
    
    query = """
        SELECT 1
        FROM user_favorites
        WHERE user_id = %s AND article_id = %s;
    """
    
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (user_id, article_id))
            result = cur.fetchone() is not None
            
            # Сохраняем в кэш на 1 час
            redis.setex(cache_key, 3600, "1" if result else "0")
            return result

def get_article_title(article_id):
    """
    Получает название статьи по ID с кэшированием в Redis.
    """
    redis = redis_client.get_connection()
    cache_key = f"article_title:{article_id}"
    
    # Пытаемся получить из кэша
    cached_title = redis.get(cache_key)
    if cached_title:
        return cached_title
    
    query = """
        SELECT title
        FROM articles
        WHERE article_id = %s;
    """
    
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (article_id,))
            result = cur.fetchone()
            title = result[0] if result else "Неизвестная статья"
            
            # Сохраняем в кэш на 1 час
            redis.setex(cache_key, 3600, title)
            return title

def get_article_id_by_title(title):
    """
    Получает ID статьи по названию с кэшированием в Redis.
    """
    redis = redis_client.get_connection()
    cache_key = f"article_id:{title}"
    
    # Пытаемся получить из кэша
    cached_id = redis.get(cache_key)
    if cached_id:
        return int(cached_id) if cached_id != "None" else None
    
    query = """
        SELECT article_id
        FROM articles
        WHERE title = %s;
    """
    
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (title,))
            result = cur.fetchone()
            article_id = result[0] if result else None
            
            # Сохраняем в кэш на 1 час
            redis.setex(cache_key, 3600, str(article_id) if article_id else "None")
            return article_id