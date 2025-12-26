@echo off
echo ========================================
echo Запуск автотестов
echo ========================================

cd tests
python test_crawler.py

echo.
echo Результаты сохранены в test_results.txt
pause