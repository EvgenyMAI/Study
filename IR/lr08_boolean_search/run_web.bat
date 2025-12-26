@echo off
echo Starting web server...
echo.
echo Web interface will be available at: http://localhost:5000
echo Press Ctrl+C to stop
echo.

cd web
python app.py
cd ..

pause