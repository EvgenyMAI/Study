@echo off
chcp 65001 >nul
echo ========================================
echo Лабораторная работа 3: Токенизация
echo ========================================

REM Шаг 1: Компиляция
echo.
echo [1/4] Компиляция C++ кода...
cd tokenizer

if not exist tokenizer.exe (
    echo Компилируем...
    g++ -std=c++17 -O3 -Wall -Wextra -c tokenizer.cpp -o tokenizer.o
    g++ -std=c++17 -O3 -Wall -Wextra -c main.cpp -o main.o
    g++ -std=c++17 -O3 -o tokenizer.exe tokenizer.o main.o
    
    if not exist tokenizer.exe (
        echo ОШИБКА: Не удалось скомпилировать!
        cd ..
        pause
        exit /b 1
    )
)

echo Токенизатор готов!

REM Шаг 2: Автотесты
echo.
echo [2/4] Запуск автотестов...

if not exist test_tokenizer.exe (
    g++ -std=c++17 -O3 -Wall -Wextra -c test_tokenizer.cpp -o test_tokenizer.o 2>nul
    g++ -std=c++17 -O3 -o test_tokenizer.exe tokenizer.o test_tokenizer.o 2>nul
)

if exist test_tokenizer.exe (
    test_tokenizer.exe
    if errorlevel 1 (
        echo ВНИМАНИЕ: Некоторые тесты не прошли
    ) else (
        echo Все тесты пройдены!
    )
) else (
    echo Тесты не найдены, пропускаем...
)

REM Шаг 3: Токенизация
echo.
echo [3/4] Токенизация документов из MongoDB...
cd ..\scripts

python run_tokenizer.py
if errorlevel 1 (
    echo ОШИБКА токенизации!
    cd ..
    pause
    exit /b 1
)

REM Шаг 4: Анализ
echo.
echo [4/4] Анализ проблемных токенов...
python analyze_tokens.py

cd ..

echo.
echo ========================================
echo Готово! Проверьте:
echo   - output/tokens.txt
echo   - output/statistics.json
echo   - tests/test_results.txt
echo ========================================
pause