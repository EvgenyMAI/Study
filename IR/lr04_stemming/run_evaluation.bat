@echo off
echo Evaluating search quality...
echo.

cd scripts
python evaluate_search.py
cd ..

echo.
echo Evaluation complete!
pause