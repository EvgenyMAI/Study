import re
import time
import streamlit as st
from repositories.articles import add_article, get_articles, update_article, delete_article

def show_admin_panel(user_id):
    st.markdown("<h1 style='text-align: center;'>Редактирование статей</h1>", unsafe_allow_html=True)

    # Добавление новой статьи
    show_add_article_form(user_id)

    # Список и управление статьями
    show_articles_list()

def show_add_article_form(user_id):
    """Отображает форму для добавления новой статьи."""
    st.markdown("<h4 style='text-align: center;'>Добавление новой статьи</h4>", unsafe_allow_html=True)

    title = st.text_input("Название статьи", key="title_input")
    publication_year = st.number_input("Год публикации", min_value=1000, max_value=3000, value=2000, step=1, key="publication_year_input")
    link = st.text_input("Ссылка на статью", key="link_input")
    authors = st.text_input("Авторы (вводите полные имена через запятую)", key="authors_input")
    keywords = st.text_input("Ключевые слова (через запятую)", key="keywords_input")

    # Проверка на корректность данных
    link_error, author_error = validate_article_input(link, authors)

    if st.button("Добавить статью", key="add_article_button"):
        if not (title and publication_year and link and authors and keywords):
            st.error("Все поля должны быть заполнены!")
        elif author_error or link_error:
            st.error("Исправьте ошибки перед добавлением статьи.")
        else:
            add_new_article(title, publication_year, link, authors, keywords, user_id)

def validate_article_input(link, authors):
    """Проверяет корректность ссылки и формата авторов."""
    link_error = False
    if link and not re.match(r'^https://', link):
        st.error("Ссылка должна начинаться с 'https://'. Убедитесь, что введенный URL корректен.")
        link_error = True

    author_error = False
    if authors and any(len(author.split()) < 2 for author in authors.split(",")):
        st.error("Некорректный формат имени авторов. Убедитесь, что каждый автор имеет формат 'Имя Фамилия'.")
        author_error = True

    return link_error, author_error

def add_new_article(title, publication_year, link, authors, keywords, user_id):
    """Добавляет статью в базу данных."""
    authors_list = [author.strip() for author in authors.split(",") if author.strip()]
    keywords_list = [keyword.strip() for keyword in keywords.split(",")]

    article_data = {
        "title": title,
        "publication_year": publication_year,
        "link": link,
        "authors": authors_list,
        "keywords": keywords_list,
        "uploaded_by": user_id
    }

    try:
        article_id = add_article(article_data)
        st.success(f"Статья успешно добавлена! ID статьи: {article_id}")
        st.rerun()
    except Exception as e:
        st.error(f"Ошибка при добавлении статьи: {str(e)}")

def show_articles_list():
    """Отображает список статей и их управление."""
    st.markdown("<h4 style='text-align: center;'>Список добавленных статей</h4>", unsafe_allow_html=True)

    try:
        articles = get_articles()
        articles["authors_str"] = articles["authors"].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
        articles["keywords_str"] = articles["keywords"].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
        articles.index += 1  # Индексация с 1

        # Фильтры
        title_filter, article_id_filter = show_filters()

        filtered_articles = apply_filters(articles, title_filter, article_id_filter)

        # Отображение статей
        display_articles(filtered_articles)

    except Exception as e:
        st.error(f"Ошибка при загрузке статей: {str(e)}")

def show_filters():
    """Отображает фильтры для поиска статей."""
    st.sidebar.header("Фильтры")
    title_filter = st.sidebar.text_input("Название статьи", key="title_filter_input")
    article_id_filter = st.sidebar.text_input("ID статьи", key="article_id_filter_input")

    return title_filter, article_id_filter

def apply_filters(articles, title_filter, article_id_filter):
    """Применяет фильтры к списку статей."""
    if title_filter:
        articles = articles[articles["title"].str.contains(title_filter, case=False, na=False)]

    if article_id_filter:
        try:
            article_id = int(article_id_filter)
            articles = articles[articles["article_id"] == article_id]
        except ValueError:
            st.sidebar.error("ID статьи должен быть числом!")

    return articles

def display_articles(articles):
    """Отображает статьи с возможностью редактирования и удаления."""
    if "editing_article_id" not in st.session_state:
        st.session_state["editing_article_id"] = None

    for idx, row in articles.iterrows():
        if st.session_state["editing_article_id"] == row["article_id"]:
            show_edit_article_form(row)
        else:
            show_article_actions(row)

def show_edit_article_form(row):
    """Отображает форму для редактирования статьи."""
    with st.form(key=f"edit_form_{row['article_id']}"):
        new_title = st.text_input("Название статьи", value=row['title'], key=f"edit_title_{row['article_id']}")
        new_publication_year = st.number_input("Год публикации", min_value=1000, max_value=3000, value=row['publication_year'], key=f"edit_year_{row['article_id']}")
        new_link = st.text_input("Ссылка на статью", value=row['link'], key=f"edit_link_{row['article_id']}")
        new_authors = st.text_input("Авторы (через запятую)", value=', '.join(row['authors']), key=f"edit_authors_{row['article_id']}")
        new_keywords = st.text_input("Ключевые слова (через запятую)", value=', '.join(row['keywords']), key=f"edit_keywords_{row['article_id']}")

        submit = st.form_submit_button("Сохранить изменения")
        cancel = st.form_submit_button("Отмена")

        if submit:
            update_article_data(row['article_id'], new_title, new_publication_year, new_link, new_authors, new_keywords)
        if cancel:
            st.session_state["editing_article_id"] = None
            st.rerun()

def update_article_data(article_id, title, publication_year, link, authors, keywords):
    """Обновляет данные статьи в базе данных."""
    try:
        updated_data = {
            "article_id": article_id,
            "title": title.strip(),
            "publication_year": publication_year,
            "link": link.strip(),
            "authors": [author.strip() for author in authors.split(',')],
            "keywords": [keyword.strip() for keyword in keywords.split(',')],
        }
        update_article(updated_data)
        st.session_state["editing_article_id"] = None
        st.success(f"Изменения сохранены для статьи ID {article_id}!")
        time.sleep(2)
        st.rerun()
    except Exception as e:
        st.error(f"Ошибка при сохранении изменений: {e}")

def show_article_actions(row):
    """Отображает кнопки редактирования и удаления статьи."""
    st.write(f"ID: {row['article_id']} | Название: {row['title']}")
    if st.button("Редактировать", key=f"edit_{row['article_id']}"):
        st.session_state["editing_article_id"] = row["article_id"]
        st.rerun()

    if st.button("Удалить", key=f"delete_{row['article_id']}"):
        delete_article_data(row['article_id'])

def delete_article_data(article_id):
    """Удаляет статью из базы данных."""
    try:
        delete_article(article_id)
        st.success(f"Статья с ID {article_id} удалена!")
        time.sleep(2)
        st.rerun()
    except Exception as e:
        st.error(f"Ошибка при удалении статьи: {e}")