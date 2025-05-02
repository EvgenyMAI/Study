import pandas as pd
import streamlit as st
import repositories.articles

def show_articles_page():
    st.markdown("<h1 style='text-align: center;'>Список статей</h1>", unsafe_allow_html=True)

    # Получение данных
    articles = repositories.articles.get_articles()

    if articles.empty:
        st.warning("Нет доступных статей.")
        return

    # Преобразование данных
    articles = preprocess_articles_data(articles)

    # Фильтры
    filtered_articles = apply_filters(articles)

    # Отображение статей в таблице
    display_articles_table(filtered_articles)

    # Открытие статей
    st.markdown("<h4 style='text-align: center;'>Открыть статью</h4>", unsafe_allow_html=True)
    display_article_links(filtered_articles)

def preprocess_articles_data(articles):
    """Подготовка данных статей для отображения."""
    articles["authors_str"] = articles["authors"].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    articles["keywords_str"] = articles["keywords"].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    articles.index += 1  # Индексация с 1
    return articles

def apply_filters(articles):
    """Применяет фильтры к статьям на основе ввода пользователя."""
    st.sidebar.header("Фильтры")
    title_filter = st.sidebar.text_input("Название статьи", key="title_filter_input")
    year_filter = st.sidebar.text_input("Год", key="year_filter_input")
    author_filter = st.sidebar.text_input("Авторы", key="author_filter_input")
    keyword_filter = st.sidebar.text_input("Ключевые слова", key="keyword_filter_input")

    filtered_articles = articles.copy()

    if author_filter:
        filtered_articles = filtered_articles[filtered_articles["authors_str"].str.contains(author_filter, case=False, na=False)]

    if year_filter:
        try:
            year = int(year_filter)
            filtered_articles = filtered_articles[filtered_articles["publication_year"] == year]
        except ValueError:
            st.sidebar.error("Год должен быть числом!")

    if keyword_filter:
        filtered_articles = filtered_articles[filtered_articles["keywords_str"].str.contains(keyword_filter, case=False, na=False)]

    if title_filter:
        filtered_articles = filtered_articles[filtered_articles["title"].str.contains(title_filter, case=False, na=False)]

    return filtered_articles

def display_articles_table(articles):
    """Отображает таблицу статей."""
    column_rename_map = {
        'title': 'Название статьи',
        'publication_year': 'Год',
        'authors': 'Авторы',
        'keywords': 'Ключевые слова'
    }
    articles_renamed = articles.rename(columns=column_rename_map)
    st.dataframe(articles_renamed[['Название статьи', 'Год', 'Авторы', 'Ключевые слова']])

def display_article_links(articles):
    """Отображает ссылки на статьи для открытия."""
    for idx, row in articles.iterrows():
        st.markdown(f"{idx}. [{row['title']}]({row['link']}) — *{row['authors']}* ({row['publication_year']})")