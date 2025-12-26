@echo off
echo ========================================
echo Компиляция токенизатора (C++)
echo ========================================

cd tokenizer

echo.
echo Проверка компилятора...
where g++ >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ОШИБКА: g++ не найден!
    echo Установите MinGW: https://sourceforge.net/projects/mingw/
    pause
    exit /b 1
)

echo.
echo Очистка старых файлов...
if exist *.o del *.o
if exist tokenizer.exe del tokenizer.exe
if exist test_tokenizer.exe del test_tokenizer.exe

echo.
echo Компиляция tokenizer.cpp...
g++ -std=c++17 -O3 -Wall -Wextra -c tokenizer.cpp -o tokenizer.o
if %ERRORLEVEL% NEQ 0 (
    echo ОШИБКА компиляции tokenizer.cpp!
    pause
    exit /b 1
)

echo.
echo Компиляция main.cpp...
g++ -std=c++17 -O3 -Wall -Wextra -c main.cpp -o main.o
if %ERRORLEVEL% NEQ 0 (
    echo ОШИБКА компиляции main.cpp!
    pause
    exit /b 1
)

echo.
echo Линковка tokenizer.exe...
g++ -std=c++17 -O3 -o tokenizer.exe tokenizer.o main.o
if %ERRORLEVEL% NEQ 0 (
    echo ОШИБКА линковки!
    pause
    exit /b 1
)

if exist tokenizer.exe (
    echo.
    echo ========================================
    echo Компиляция tokenizer успешна!
    echo Создан: tokenizer.exe
    echo ========================================
) else (
    echo.
    echo ОШИБКА: tokenizer.exe не создан!
    pause
    exit /b 1
)

echo.
echo Компиляция тестов...
g++ -std=c++17 -O3 -Wall -Wextra -c test_tokenizer.cpp -o test_tokenizer.o
if %ERRORLEVEL% NEQ 0 (
    echo ОШИБКА компиляции тестов!
    pause
    exit /b 1
)

g++ -std=c++17 -O3 -o test_tokenizer.exe tokenizer.o test_tokenizer.o
if %ERRORLEVEL% NEQ 0 (
    echo ОШИБКА линковки тестов!
    pause
    exit /b 1
)

if exist test_tokenizer.exe (
    echo Создан: test_tokenizer.exe
    echo.
    echo ========================================
    echo Все программы скомпилированы!
    echo ========================================
)

cd ..
echo.
pause