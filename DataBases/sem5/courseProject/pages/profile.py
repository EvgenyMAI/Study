import pandas as pd
import streamlit as st
from repositories.profile import delete_user_account

def show_profile_page(user_data):
    """Отображение профиля пользователя."""
    # Проверка на авторизацию
    if not st.session_state.get("authenticated") or not st.session_state.get("user"):
        st.error("Вы не авторизованы. Пожалуйста, войдите в систему.")
        return

    st.markdown("<h1 style='text-align: center;'>Профиль</h1>", unsafe_allow_html=True)

    # Проверка на наличие данных пользователя
    if user_data is None or user_data.empty:
        st.error("Ошибка: данные пользователя отсутствуют.")
        return

    # Определяем формат user_data (DataFrame или dict)
    if isinstance(user_data, pd.DataFrame):
        user = user_data.iloc[0]
    elif isinstance(user_data, dict):
        user = user_data
    else:
        st.error("Ошибка: некорректный формат данных пользователя.")
        return

    # Отображение личной информации
    st.markdown("<h2>Личная информация</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])  # Колонки для упорядочивания
    with col1:
        st.write(f"**Имя:** {user.get('first_name', 'Не указано')} {user.get('last_name', 'Не указано')}")
        st.write(f"**Email:** {user.get('email', 'Не указано')}")
        st.write(f"**Роль:** {'Администратор' if user.get('role') == 'admin' else 'Пользователь'}")
        st.write(f"**ID пользователя:** {user.get('user_id', 'Не указано')}")
        st.write(f"**Дата регистрации:** {user.get('registration_date', 'Не указано')}")

    if user.get("role") == "user":
        st.info("Если вы хотите стать администратором, обратитесь к текущему администратору.")

    st.markdown("<hr>", unsafe_allow_html=True)  # Разделительная линия

    # Раздел: Управление аккаунтом
    st.markdown("<h3>Управление аккаунтом</h3>", unsafe_allow_html=True)

    # Кнопка для выхода из аккаунта
    if st.button("Выйти из аккаунта"):
        logout_user()

    # Раздел: Удаление аккаунта
    st.markdown("<h3>Удаление аккаунта</h3>", unsafe_allow_html=True)
    confirm_delete = st.text_input("Введите 'удалить', чтобы подтвердить удаление аккаунта", key="delete_confirm")

    if st.button("Подтвердить удаление"):
        if confirm_delete.strip().lower() == "удалить":
            delete_account(user)
        else:
            st.warning("Введите точное подтверждение: 'удалить'.")

def logout_user():
    """Выход из аккаунта и сброс состояния сессии."""
    st.session_state.clear()
    st.success("Вы вышли из аккаунта. Возврат на страницу авторизации...")
    st.rerun()

def delete_account(user):
    """Удаление аккаунта пользователя."""
    user_id = user.get("user_id")
    if not user_id:
        st.error("Ошибка: некорректные данные пользователя для удаления.")
        return

    if delete_user_account(int(user_id)):
        st.success("Ваш аккаунт был успешно удален!")
        st.session_state.clear()
        st.rerun()
    else:
        st.error("Ошибка при удалении аккаунта.")