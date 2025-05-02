-- Добавление пользователей
INSERT INTO users (email, password, first_name, last_name, role) VALUES
('admin@example.com', 'qwe1', 'Евгений', 'Кострюков', 'admin'),
('user1@example.com', 'qwe1', 'Артемий', 'Лебедев', 'user'),
('user2@example.com', 'qwe1', 'Борис', 'Бритва', 'user');

-- Добавление авторов
INSERT INTO authors (first_name, last_name) VALUES
('Иванов', 'Иван'),
('Александр', 'Пушкин'),
('Андрей', 'Малахов'),
('Федор', 'Бондарчук');

-- Добавление статей
INSERT INTO articles (title, publication_year, link, uploaded_by) VALUES
('Исследование квантовой запутанности', 2024, 'https://example.com/quantum_entanglement', 1),
('Анализ больших данных в медицине', 2023, 'https://example.com/big_data_medicine', 1),
('Технологии искусственного интеллекта в образовании', 2022, 'https://example.com/ai_education', 1);

-- Связь статей с авторами
INSERT INTO article_authors (article_id, author_id) VALUES
(1, 1), (1, 2),
(2, 3), (2, 4),
(3, 1), (3, 3);

-- Добавление ключевых слов
INSERT INTO keywords (keyword) VALUES
('Квантовая физика'),
('Большие данные'),
('Медицина'),
('Искусственный интеллект'),
('Образование');

-- Связь статей с ключевыми словами
INSERT INTO article_keywords (article_id, keyword_id) VALUES
(1, 1),
(2, 2), (2, 3),
(3, 4), (3, 5);

-- Добавление избранных статей пользователей
INSERT INTO user_favorites (user_id, article_id) VALUES
(1, 1), (1, 2), (2, 3),
(3, 1);