# Лабораторные работы по компьютерной графике
## Руководство по настройке и запуску
### 1. Создать docker-image:
```
docker-compose --env-file env.env up -d
```
### 2. Установить необходимые компоненты:
```
pip install -r requirements.txt
```
Работает с версией Python 3.12.3
### 3. Установить модуль bcrypt:
```
pip install bcrypt
```
### 4. В редакторе баз данных (например, **DBeaver**) выполнить следующие скрипты для поднятой бд: 
- Cкрипт по созданию таблиц и связей между ними (код скрипта находится в файле **migrations/ddl.sql**);
- Скрипт по заполнению таблиц первоначальными данными (код скрипта находится в файле **migrations/dml.sql**)
### 5. В любой среде выполнить скрипт **passw.py**, который захеширует все незахешированные пароли.
### 6. Запустить проект:
```
streamlit run main.py
```