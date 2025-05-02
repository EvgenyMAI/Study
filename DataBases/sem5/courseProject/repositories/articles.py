import psycopg2
import psycopg2.extras
from settings import DB_CONFIG
import pandas as pd

def cleanup_unused_entries():
    """
    Удаляет неиспользуемых авторов и ключевые слова из базы данных.
    """
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            # Удаляем неиспользуемых авторов
            query_delete_orphan_authors = """
                DELETE FROM authors
                WHERE author_id NOT IN (SELECT DISTINCT author_id FROM article_authors);
            """
            cur.execute(query_delete_orphan_authors)

            # Удаляем неиспользуемые ключевые слова
            query_delete_orphan_keywords = """
                DELETE FROM keywords
                WHERE keyword_id NOT IN (SELECT DISTINCT keyword_id FROM article_keywords);
            """
            cur.execute(query_delete_orphan_keywords)
        conn.commit()

import psycopg2

def add_article(article_data):
    """
    Добавляет статью в базу данных.

    :param article_data: Словарь с данными статьи (title, publication_year, link, uploaded_by, authors, keywords).
    :return: ID добавленной статьи.
    """
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            # Добавление статьи
            query_article = """
                INSERT INTO articles (title, publication_year, link, uploaded_by)
                VALUES (%(title)s, %(publication_year)s, %(link)s, %(uploaded_by)s)
                RETURNING article_id;
            """
            cur.execute(query_article, article_data)
            article_id = cur.fetchone()[0]

            # Обработка авторов
            if article_data.get('authors'):
                for author_name in article_data['authors']:
                    # Проверяем корректность имени и фамилии
                    name_parts = author_name.split()
                    if len(name_parts) < 2:
                        raise ValueError(f"Некорректный формат имени автора: '{author_name}'. Ожидается формат 'Имя Фамилия'.")

                    first_name, last_name = name_parts[:2]  # Берем только первые два слова (Имя и Фамилия)

                    # Проверяем, существует ли автор
                    query_check_author = """
                        SELECT author_id FROM authors
                        WHERE first_name = %(first_name)s AND last_name = %(last_name)s;
                    """
                    cur.execute(query_check_author, {"first_name": first_name, "last_name": last_name})
                    result = cur.fetchone()

                    # Если автор отсутствует, добавляем его
                    if result is None:
                        query_add_author = """
                            INSERT INTO authors (first_name, last_name)
                            VALUES (%(first_name)s, %(last_name)s)
                            RETURNING author_id;
                        """
                        cur.execute(query_add_author, {"first_name": first_name, "last_name": last_name})
                        author_id = cur.fetchone()[0]
                    else:
                        author_id = result[0]

                    # Связываем автора со статьей
                    query_link_author = """
                        INSERT INTO article_authors (article_id, author_id)
                        VALUES (%s, %s);
                    """
                    cur.execute(query_link_author, (article_id, author_id))

            # Обработка ключевых слов
            if article_data.get('keywords'):
                for keyword in article_data['keywords']:
                    # Проверяем, существует ли ключевое слово
                    query_check_keyword = """
                        SELECT keyword_id FROM keywords
                        WHERE keyword = %s;
                    """
                    cur.execute(query_check_keyword, (keyword,))
                    result = cur.fetchone()

                    # Если ключевое слово отсутствует, добавляем его
                    if result is None:
                        query_add_keyword = """
                            INSERT INTO keywords (keyword)
                            VALUES (%s)
                            RETURNING keyword_id;
                        """
                        cur.execute(query_add_keyword, (keyword,))
                        keyword_id = cur.fetchone()[0]
                    else:
                        keyword_id = result[0]

                    # Связываем ключевое слово со статьей
                    query_link_keyword = """
                        INSERT INTO article_keywords (article_id, keyword_id)
                        VALUES (%s, %s);
                    """
                    cur.execute(query_link_keyword, (article_id, keyword_id))

            return article_id

def get_articles():
    query = """
        SELECT a.article_id, a.title, 
               array_agg(DISTINCT CONCAT(au.first_name, ' ', au.last_name)) AS authors, 
               a.publication_year, a.link,
               CONCAT(u.first_name, ' ', u.last_name) AS user_name,
               array_agg(DISTINCT k.keyword) AS keywords  -- Добавляем агрегацию для ключевых слов
        FROM articles a
        LEFT JOIN article_authors aa ON a.article_id = aa.article_id
        LEFT JOIN authors au ON aa.author_id = au.author_id
        LEFT JOIN users u ON a.uploaded_by = u.user_id
        LEFT JOIN article_keywords ak ON a.article_id = ak.article_id
        LEFT JOIN keywords k ON ak.keyword_id = k.keyword_id
        GROUP BY a.article_id, u.first_name, u.last_name
        ORDER BY a.publication_year DESC;
    """
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            # Преобразуем результат в DataFrame
            articles = cur.fetchall()
            columns = ['article_id', 'title', 'authors', 'publication_year', 'link', 'user_name', 'keywords']
            return pd.DataFrame(articles, columns=columns)

def update_article(article_data):
    """
    Обновляет данные статьи в базе данных.

    :param article_data: Словарь с обновленными данными статьи.
                         Ожидается: article_id, title, publication_year, link, authors, keywords.
    """
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            try:
                # Обновление основной информации о статье
                query_update_article = """
                    UPDATE articles
                    SET title = %(title)s,
                        publication_year = %(publication_year)s,
                        link = %(link)s
                    WHERE article_id = %(article_id)s;
                """
                cur.execute(query_update_article, article_data)

                # Удаление существующих связей
                cur.execute("DELETE FROM article_authors WHERE article_id = %s;", (article_data['article_id'],))
                cur.execute("DELETE FROM article_keywords WHERE article_id = %s;", (article_data['article_id'],))

                # Обновление авторов
                if article_data.get('authors'):
                    for author_name in article_data['authors']:
                        name_parts = author_name.strip().split()
                        if len(name_parts) < 2:
                            raise ValueError(f"Некорректный формат имени автора: '{author_name}'. Ожидается формат 'Имя Фамилия'.")
                        first_name, last_name = name_parts[:2]

                        # Проверяем или добавляем автора
                        query_check_author = """
                            SELECT author_id FROM authors
                            WHERE first_name = %(first_name)s AND last_name = %(last_name)s;
                        """
                        cur.execute(query_check_author, {"first_name": first_name, "last_name": last_name})
                        result = cur.fetchone()
                        if result is None:
                            query_add_author = """
                                INSERT INTO authors (first_name, last_name)
                                VALUES (%(first_name)s, %(last_name)s)
                                RETURNING author_id;
                            """
                            cur.execute(query_add_author, {"first_name": first_name, "last_name": last_name})
                            author_id = cur.fetchone()[0]
                        else:
                            author_id = result[0]

                        # Связываем автора со статьей
                        query_link_author = """
                            INSERT INTO article_authors (article_id, author_id)
                            VALUES (%s, %s);
                        """
                        cur.execute(query_link_author, (article_data['article_id'], author_id))

                # Обновление ключевых слов
                if article_data.get('keywords'):
                    for keyword in article_data['keywords']:
                        keyword = keyword.strip()
                        query_check_keyword = "SELECT keyword_id FROM keywords WHERE keyword = %s;"
                        cur.execute(query_check_keyword, (keyword,))
                        result = cur.fetchone()
                        if result is None:
                            query_add_keyword = """
                                INSERT INTO keywords (keyword)
                                VALUES (%s)
                                RETURNING keyword_id;
                            """
                            cur.execute(query_add_keyword, (keyword,))
                            keyword_id = cur.fetchone()[0]
                        else:
                            keyword_id = result[0]

                        # Связываем ключевое слово со статьей
                        query_link_keyword = """
                            INSERT INTO article_keywords (article_id, keyword_id)
                            VALUES (%s, %s);
                        """
                        cur.execute(query_link_keyword, (article_data['article_id'], keyword_id))

                # Коммит изменений
                conn.commit()

                # Очищаем неиспользуемые записи
                cleanup_unused_entries()

            except Exception as e:
                conn.rollback()
                raise RuntimeError(f"Ошибка при обновлении статьи: {str(e)}")

def delete_article(article_id):
    """
    Удаляет статью из базы данных вместе с неиспользуемыми данными (авторы и ключевые слова).

    :param article_id: ID статьи.
    """
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            # Удаляем связи статьи с авторами и ключевыми словами
            query_delete_authors = "DELETE FROM article_authors WHERE article_id = %s;"
            cur.execute(query_delete_authors, (article_id,))

            query_delete_keywords = "DELETE FROM article_keywords WHERE article_id = %s;"
            cur.execute(query_delete_keywords, (article_id,))

            # Удаляем саму статью
            query_delete_article = "DELETE FROM articles WHERE article_id = %s;"
            cur.execute(query_delete_article, (article_id,))

        conn.commit()

    # Очищаем неиспользуемые записи
    cleanup_unused_entries()