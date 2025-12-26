@echo off
echo Preparing index for boolean search...
echo.

cd scripts
python build_index.py
cd ..

echo.
echo Index preparation complete!
pause