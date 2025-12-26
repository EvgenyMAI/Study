@echo off
echo Building boolean index...

if not exist build mkdir build

g++ -o build/boolean_index.exe src/posting_list.cpp src/boolean_index.cpp src/boolean_query.cpp src/main.cpp -static-libgcc -static-libstdc++

if %ERRORLEVEL% EQU 0 (
    echo Build successful!
) else (
    echo Build failed!
    exit /b 1
)

echo.
echo Building tests...

g++ -o build/test.exe src/posting_list.cpp src/boolean_index.cpp src/boolean_query.cpp src/test.cpp -static-libgcc -static-libstdc++

if %ERRORLEVEL% EQU 0 (
    echo Tests build successful!
) else (
    echo Tests build failed!
    exit /b 1
)

echo.
echo Done!