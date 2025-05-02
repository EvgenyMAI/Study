-- Удаление связи между статьями и ключевыми словами
DROP TABLE IF EXISTS article_keywords;

-- Удаление связи между статьями и авторами
DROP TABLE IF EXISTS article_authors;

-- Удаление таблицы избранных статей пользователей
DROP TABLE IF EXISTS user_favorites;

-- Удаление таблицы статей
DROP TABLE IF EXISTS articles;

-- Удаление таблицы ключевых слов
DROP TABLE IF EXISTS keywords;

-- Удаление таблицы авторов
DROP TABLE IF EXISTS authors;

-- Удаление таблицы пользователей
DROP TABLE IF EXISTS users;

-- Удаление типа user_role
DROP TYPE IF EXISTS user_role;

-- Тип для ролей пользователей
CREATE TYPE user_role AS ENUM ('user', 'admin');

-- Таблица пользователей
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password TEXT NOT NULL,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    role user_role NOT NULL DEFAULT 'user'
);

COMMENT ON TABLE users IS 'Информация о пользователях';
COMMENT ON COLUMN users.user_id IS 'Уникальный идентификатор пользователя';
COMMENT ON COLUMN users.email IS 'Email пользователя';
COMMENT ON COLUMN users.password IS 'Пароль пользователя';
COMMENT ON COLUMN users.first_name IS 'Имя пользователя';
COMMENT ON COLUMN users.last_name IS 'Фамилия пользователя';
COMMENT ON COLUMN users.registration_date IS 'Дата регистрации пользователя';
COMMENT ON COLUMN users.role IS 'Роль пользователя (user или admin)';

-- Таблица авторов статей
CREATE TABLE authors (
    author_id SERIAL PRIMARY KEY,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL
);

COMMENT ON TABLE authors IS 'Информация об авторах статей';
COMMENT ON COLUMN authors.author_id IS 'Уникальный идентификатор автора';
COMMENT ON COLUMN authors.first_name IS 'Имя автора';
COMMENT ON COLUMN authors.last_name IS 'Фамилия автора';

-- Таблица научных статей
CREATE TABLE articles (
    article_id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    publication_year INT NOT NULL,
    link TEXT,
    uploaded_by INT REFERENCES users(user_id) ON DELETE SET NULL
);

COMMENT ON TABLE articles IS 'Информация о научных статьях';
COMMENT ON COLUMN articles.article_id IS 'Уникальный идентификатор статьи';
COMMENT ON COLUMN articles.title IS 'Название статьи';
COMMENT ON COLUMN articles.publication_year IS 'Год публикации';
COMMENT ON COLUMN articles.link IS 'Ссылка на статью';
COMMENT ON COLUMN articles.uploaded_by IS 'Идентификатор пользователя, загрузившего статью';

-- Таблица связей между статьями и авторами
CREATE TABLE article_authors (
    article_id INT REFERENCES articles(article_id) ON DELETE CASCADE,
    author_id INT REFERENCES authors(author_id) ON DELETE CASCADE,
    PRIMARY KEY (article_id, author_id)
);

COMMENT ON TABLE article_authors IS 'Связь между статьями и их авторами';
COMMENT ON COLUMN article_authors.article_id IS 'Идентификатор статьи';
COMMENT ON COLUMN article_authors.author_id IS 'Идентификатор автора';

-- Таблица ключевых слов
CREATE TABLE keywords (
    keyword_id SERIAL PRIMARY KEY,
    keyword VARCHAR(100) UNIQUE NOT NULL
);

COMMENT ON TABLE keywords IS 'Список ключевых слов';
COMMENT ON COLUMN keywords.keyword_id IS 'Уникальный идентификатор ключевого слова';
COMMENT ON COLUMN keywords.keyword IS 'Ключевое слово';

-- Таблица связей между статьями и ключевыми словами
CREATE TABLE article_keywords (
    article_id INT REFERENCES articles(article_id) ON DELETE CASCADE,
    keyword_id INT REFERENCES keywords(keyword_id) ON DELETE CASCADE,
    PRIMARY KEY (article_id, keyword_id)
);

COMMENT ON TABLE article_keywords IS 'Связь между статьями и их ключевыми словами';
COMMENT ON COLUMN article_keywords.article_id IS 'Идентификатор статьи';
COMMENT ON COLUMN article_keywords.keyword_id IS 'Идентификатор ключевого слова';

-- Таблица связи между пользователями и их избранными статьями
CREATE TABLE user_favorites (
    user_id INT REFERENCES users(user_id) ON DELETE CASCADE,
    article_id INT REFERENCES articles(article_id) ON DELETE CASCADE,
    PRIMARY KEY (user_id, article_id)
);

COMMENT ON TABLE user_favorites IS 'Связь между пользователями и их избранными статьями';
COMMENT ON COLUMN user_favorites.user_id IS 'Идентификатор пользователя';
COMMENT ON COLUMN user_favorites.article_id IS 'Идентификатор статьи';