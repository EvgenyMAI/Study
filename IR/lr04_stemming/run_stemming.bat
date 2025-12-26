@echo off
echo Applying stemming to corpus...
echo.

cd scripts
python apply_stemming.py
cd ..

echo.
echo Stemming complete!
pause