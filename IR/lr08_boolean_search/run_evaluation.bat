@echo off
echo Evaluating search quality...
echo.

cd scripts
python evaluate_quality.py
cd ..

echo.
echo Evaluation complete!
pause