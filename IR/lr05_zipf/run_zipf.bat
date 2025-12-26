@echo off
chcp 65001 >nul
echo ========================================
echo Лабораторная работа 5: Закон Ципфа
echo ========================================

echo.
echo [1/3] Подсчёт частот терминов...
cd scripts
python calculate_frequencies.py

echo.
echo [2/3] Построение графика...
python plot_zipf.py

echo.
echo [3/3] Анализ отклонений...
python analyze_zipf.py

cd ..

echo.
echo ========================================
echo Готово! Проверьте:
echo   - output/term_frequencies.json
echo   - output/zipf_plot.png
echo   - output/zipf_analysis.txt
echo ========================================
pause