@echo off
echo Building boolean search engine...

if not exist build mkdir build

g++ -o build/boolean_search.exe src/search_engine.cpp src/boolean_parser.cpp src/main.cpp ../lr07_boolean_index/src/posting_list.cpp ../lr07_boolean_index/src/boolean_index.cpp -I../lr07_boolean_index/src -static-libgcc -static-libstdc++

if %ERRORLEVEL% EQU 0 (
    echo Build successful!
) else (
    echo Build failed!
    exit /b 1
)

echo.
echo Building tests...

g++ -o build/test.exe src/search_engine.cpp src/boolean_parser.cpp src/test.cpp ../lr07_boolean_index/src/posting_list.cpp ../lr07_boolean_index/src/boolean_index.cpp -I../lr07_boolean_index/src -static-libgcc -static-libstdc++

if %ERRORLEVEL% EQU 0 (
    echo Tests build successful!
) else (
    echo Tests build failed!
    exit /b 1
)

echo.
echo Done!