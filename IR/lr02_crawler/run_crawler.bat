@echo off
echo ========================================
echo Лабораторная работа 2: Поисковый робот
echo ========================================

cd crawler

echo.
echo Проверка MongoDB...
timeout /t 2 >nul

echo.
echo Установка зависимостей...
pip install -r requirements.txt

echo.
echo ========================================
echo Запуск поискового робота
echo ========================================
python crawler.py config.yaml

echo.
echo ========================================
echo Готово! Проверьте logs/crawler.log
echo ========================================
pause