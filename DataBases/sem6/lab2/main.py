import time
import pandas as pd
import business.user
import business.regist
import streamlit as st
import repositories.admin
from business.auth import Authorize
from pages.profile import show_profile_page
from pages.articles import show_articles_page
from pages.admin_panel import show_admin_panel
from pages.favorites import show_favorites_page

auth = Authorize()
registr = business.regist.Registration()

def login():
    st.title("Авторизация")
    st.write("Введите почту и пароль:")

    email = st.text_input("Почта")
    password = st.text_input("Пароль", type="password")

    if st.button("Войти"):
        try:
            # Проверка, существует ли пользователь с таким email
            user_data = business.user.get_user(email)
            if user_data is None or user_data.empty:
                st.error("Пользователь с таким email не найден.")
                return

            # Проверка пароля
            if auth.auth(email, password):
                # Генерация токена
                token = auth.generate_token(email)
                
                st.session_state["authenticated"] = True
                st.session_state["username"] = email
                st.session_state["token"] = token
                
                user_data = user_data.iloc[0]  # Получаем данные пользователя
                st.session_state["user"] = user_data.to_dict()
                st.session_state["admin"] = repositories.admin.get_admins(int(user_data["user_id"]))
                st.success(f"Добро пожаловать, {user_data['first_name']} {user_data['last_name']}!")
                time.sleep(1.0)
                st.rerun()
            else:
                st.error("Неверный пароль.")
        except Exception as e:
            st.error(f"Произошла ошибка: {str(e)}")

def register():
    st.title("Регистрация")
    st.write("Введите данные для регистрации:")

    first_name = st.text_input("Имя")
    last_name = st.text_input("Фамилия")
    email = st.text_input("Почта")
    password = st.text_input("Пароль", type="password")
    second_password = st.text_input("Подтверждение пароля", type="password")

    if st.button("Зарегистрироваться"):
        if not (first_name and last_name and email and password and second_password):
            st.error("Заполните все поля!")
        elif password != second_password:
            st.error("Пароли не совпадают!")
        else:
            try:
                user_data = pd.DataFrame({
                    "first_name": [first_name],
                    "last_name": [last_name],
                    "email": [email],
                    "password": [password],
                    "role": ["user"]
                })
                user_id = registr.registr(user_data)
                
                # Генерация токена для нового пользователя
                token = auth.generate_token(email)
                
                st.success("Успешная регистрация!")
                st.session_state["authenticated"] = True
                st.session_state["token"] = token
                st.session_state["user"] = {
                    "user_id": user_id,
                    "first_name": first_name,
                    "last_name": last_name,
                    "email": email,
                    "role": "user"
                }
                st.rerun()
            except Exception as e:
                st.error(f"Ошибка регистрации: {str(e)}")

def main():
    # Проверка токена при загрузке страницы
    if "token" in st.session_state and auth.validate_token(st.session_state["token"]):
        st.session_state["authenticated"] = True
    else:
        if "token" in st.session_state:
            del st.session_state["token"]
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        pg = st.radio("Войдите или зарегистрируйтесь", ["Вход", "Регистрация"])
        if pg == "Вход":
            login()
        elif pg == "Регистрация":
            register()
    else:
        email = st.session_state["user"]["email"]
        user_role = "Администратор" if st.session_state["admin"] else "Пользователь"
        st.sidebar.write(f"Вы вошли как: {user_role}")

        # Получаем данные пользователя по email
        user_data = business.user.get_user(email)

        if st.session_state["admin"]:
            page = st.sidebar.radio(
                "Перейти к странице",
                ["Список статей", "Избранные статьи", "Редактирование статей", "Профиль"],
            )

            if page == "Список статей":
                show_articles_page()
            elif page == "Избранные статьи":
                show_favorites_page()
            elif page == "Редактирование статей":
                show_admin_panel(st.session_state["user"]["user_id"])
            elif page == "Профиль":
                show_profile_page(user_data)

        else:
            page = st.sidebar.radio(
                "Перейти к странице",
                ["Список статей", "Избранные статьи", "Профиль"],
            )

            if page == "Список статей":
                show_articles_page()
            elif page == "Избранные статьи":
                show_favorites_page()
            elif page == "Профиль":
                show_profile_page(user_data)

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if "admin" not in st.session_state:
    st.session_state["admin"] = False

if "user" not in st.session_state:
    st.session_state["user"] = {}

if __name__ == "__main__":
    main()