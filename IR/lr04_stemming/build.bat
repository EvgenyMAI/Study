@echo off
echo Building stemmer...

if not exist build mkdir build

g++ -o build/stemmer.exe src/stemmer.cpp src/search.cpp src/main.cpp -static-libgcc -static-libstdc++

if %ERRORLEVEL% EQU 0 (
    echo Build successful!
) else (
    echo Build failed!
    exit /b 1
)

echo.
echo Building tests...

g++ -o build/test.exe src/stemmer.cpp src/search.cpp src/test.cpp -static-libgcc -static-libstdc++

if %ERRORLEVEL% EQU 0 (
    echo Tests build successful!
) else (
    echo Tests build failed!
    exit /b 1
)

echo.
echo Building interactive search...

g++ -o build/interactive.exe src/stemmer.cpp src/search.cpp src/interactive.cpp -static-libgcc -static-libstdc++

if %ERRORLEVEL% EQU 0 (
    echo Interactive search build successful!
) else (
    echo Interactive search build failed!
    exit /b 1
)

echo.
echo Done!