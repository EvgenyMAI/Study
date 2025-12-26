@echo off
echo Building boolean index from corpus...
echo.

cd scripts
python build_index.py
cd ..

echo.
echo Index building complete!
pause