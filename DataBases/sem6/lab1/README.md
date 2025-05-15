# Инструкция по работе с проектом

## 1. Поднятие базы данных и заполнение её данными

### 1.1. Создание Docker-образа PostgreSQL с установленным расширением `pgbigm`

Для запуска контейнера с базой данных выполните следующую команду:

```bash
docker-compose up --build
```

### 1.2. Скачивание датасета с Kaggle

#### 1.2.1. Ссылка на датасет, использованный в проекте:

```
(https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store)
```

Вы также можете использовать любой другой датасет, соответствующий требованиям лабораторной работы.

#### 1.2.2. Подготовка датасета:

* Распакуйте скачанный архив.
* Поместите файл с данными в папку `dataset/`.
* Переименуйте файл в `downloaded_dataset.csv`.

### 1.3. Генерация пользователей

Для генерации пользовательских данных выполните:

```bash
python dataset_combinating_scripts/generate_users.py
```

### 1.4. Заполнение базы данных

Для импорта данных в базу данных выполните:

```bash
python dataset_combinating_scripts/fill_db.py
```

## 2. Работа с индексами

### 2.1. Создание индексов

* Откройте файл `cmd/p1/indexes.py`.
* Выберите мод `create-indexes` (просто записав его в `default`).
* Из корня проекта выполните команду:
    ```
    python -m cmd.p1.indexes
    ```
### 2.2. Запуск бенчмарков:

Без использования индексов:
* В файле `cmd/p1/indexes.py` выберите режим `benchmark-noindex`.
* Из корня проекта выполните команду:
    ```
    python -m cmd.p1.indexes
    ```

С использованием индексов:
* В файле `cmd/p1/indexes.py` выберите режим `benchmark-index`.
* * Из корня проекта выполните команду:
    ```
    python -m cmd.p1.indexes
    ```
### 2.3. Изменение запросов

SQL-запросы можно изменять в `internal/db/store_1.py`. Например:

```
-- Было:
SELECT COUNT(*) FROM products WHERE price BETWEEN 250 AND 500;

-- Можно заменить на:
SELECT COUNT(*) FROM products WHERE price BETWEEN 300 AND 700;
```

Или изменять диапазон дат:

```
SELECT COUNT(*) FROM orders WHERE order_date BETWEEN '2019-11-03' AND '2019-11-04';
```