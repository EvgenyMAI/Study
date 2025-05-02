import streamlit as st
import pandas as pd
from repositories.favorites import (
    get_user_favorites,
    add_to_favorites,
    remove_from_favorites,
    get_article_id_by_title,
    is_article_in_favorites,
)
from repositories.articles import get_articles

def show_favorites_page():
    st.markdown("<h1 style='text-align: center;'>Избранные статьи</h1>", unsafe_allow_html=True)

    # Получаем ID пользователя
    user_id = st.session_state["user"].get("user_id")
    if not user_id:
        st.error("Ошибка: данные пользователя отсутствуют.")
        return

    # Получение данных
    favorites = get_user_favorites(user_id)

    # Проверка на наличие избранных статей
    if favorites.empty:
        st.warning("У вас нет избранных статей.")
    else:
        favorites = preprocess_favorites_data(favorites)

        # Фильтры
        filtered_favorites = apply_filters(favorites)

        # Показ фильтрованных данных
        display_favorites_table(filtered_favorites)

        # Опции удаления статей
        st.markdown("<h4 style='text-align: center;'>Управление избранными статьями</h4>", unsafe_allow_html=True)
        display_favorites_actions(filtered_favorites, user_id)

    # Добавление статей в избранное
    st.markdown("<h4 style='text-align: center;'>Добавить статью в избранное</h4>", unsafe_allow_html=True)
    add_article_to_favorites(user_id)

def preprocess_favorites_data(favorites):
    """Подготавливает данные избранных статей."""
    favorites["authors_str"] = favorites["authors"].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
    favorites["keywords_str"] = favorites["keywords"].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
    favorites.index += 1  # Индексация с 1
    return favorites

def apply_filters(favorites):
    """Применяет фильтры к избранным статьям."""
    st.sidebar.header("Фильтры")
    title_filter = st.sidebar.text_input("Название статьи", key="title_filter_input")
    year_filter = st.sidebar.text_input("Год", key="year_filter_input")
    author_filter = st.sidebar.text_input("Авторы", key="author_filter_input")
    keyword_filter = st.sidebar.text_input("Ключевые слова", key="keyword_filter_input")

    filtered_favorites = favorites.copy()

    if author_filter:
        filtered_favorites = filtered_favorites[
            filtered_favorites["authors_str"].str.contains(author_filter, case=False, na=False)
        ]

    if year_filter:
        try:
            year = int(year_filter)
            filtered_favorites = filtered_favorites[filtered_favorites["publication_year"] == year]
        except ValueError:
            st.sidebar.error("Год должен быть числом!")

    if keyword_filter:
        filtered_favorites = filtered_favorites[
            filtered_favorites["keywords_str"].str.contains(keyword_filter, case=False, na=False)
        ]

    if title_filter:
        filtered_favorites = filtered_favorites[
            filtered_favorites["title"].str.contains(title_filter, case=False, na=False)
        ]

    return filtered_favorites

def display_favorites_table(favorites):
    """Отображает таблицу избранных статей."""
    column_rename_map = {
        "title": "Название статьи",
        "publication_year": "Год",
        "authors": "Авторы",
        "keywords": "Ключевые слова",
    }
    favorites_renamed = favorites.rename(columns=column_rename_map)
    st.dataframe(favorites_renamed[["Название статьи", "Год", "Авторы", "Ключевые слова"]])

def display_favorites_actions(favorites, user_id):
    """Отображает действия по управлению избранными статьями."""
    for idx, row in favorites.iterrows():
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"{idx}. [{row['title']}]({row['link']}) — *{row['authors']}* ({row['publication_year']})")
        with col2:
            if st.button("Удалить", key=f"remove_{row['article_id']}"):
                remove_from_favorites(user_id, row["article_id"])
                st.success(f"Статья '{row['title']}' удалена из избранного.")
                st.rerun()

def add_article_to_favorites(user_id):
    """Добавляет статью в избранное."""
    articles = get_articles()
    articles_titles = articles["title"].tolist()

    selected_title = st.selectbox("Выберите статью по названию", [""] + articles_titles)

    if st.button("Добавить в избранное"):
        if selected_title:
            article_id = get_article_id_by_title(selected_title)
            if article_id is None:
                st.error(f"Статья с названием '{selected_title}' не найдена.")
            elif is_article_in_favorites(user_id, article_id):
                st.warning(f"Статья '{selected_title}' уже добавлена в избранное.")
            else:
                add_to_favorites(user_id, article_id)
                st.success(f"Статья '{selected_title}' добавлена в избранное!")
                st.rerun()
        else:
            st.warning("Выберите статью из списка!")