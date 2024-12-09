# Лабораторные работы по компьютерной графике
### Ссылка на статью "C/C++ for Visual Studio Code" от Microsoft по устанвке C++ на Windows:
```
https://code.visualstudio.com/docs/languages/cpp
```
### Установка библиотеки *freeglut* через среду MSYS2 MSYS:
```
pacman -S mingw-w64-ucrt-x86_64-freeglut
```
### Установка библиотеки *glew* через среду MSYS2 MSYS:
```
pacman -S mingw-w64-ucrt-x86_64-glew
```
### Установка библиотеки *glm*
```
pacman -S mingw-w64-ucrt-x86_64-glm
```
### Установка библиотеки *glfw*
```
pacman -S mingw-w64-ucrt-x86_64-glfw
```
### Способ выполнить компиляцию и запуск программы с помощью собственной комбинации клавиш (для Visual Studio Code)
1. Откройте файл *keybindings.json*:
    - Нажмите *Ctrl+Shift+P*, введите `Preferences: Open Keyboard Shortcuts (JSON)` и выберите этот пункт.
2. Добавьте настройку для запуска задачи:
```
[
    {
        "key": "ctrl+shift+r", // или любое удобное сочетание
        "command": "workbench.action.tasks.runTask",
        "args": "Run"
    }
]
```