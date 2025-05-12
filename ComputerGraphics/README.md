# Лабораторные работы по компьютерной графике
### Ссылка на статью "C/C++ for Visual Studio Code" от Microsoft по устанвке C++ на Windows:
```
https://code.visualstudio.com/docs/languages/cpp
```
### Установка необходимых библиотек через среду MSYS2 MSYS
**freeglut**:
```
pacman -S mingw-w64-ucrt-x86_64-freeglut
```
**glew**:
```
pacman -S mingw-w64-ucrt-x86_64-glew
```
**glm**:
```
pacman -S mingw-w64-ucrt-x86_64-glm
```
**glfw**:
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
