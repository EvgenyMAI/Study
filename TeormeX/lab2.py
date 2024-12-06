import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

Steps = 1001
t_fin = 20
t = np.linspace(0, t_fin, Steps) # Равномерно распределяем Steps элементов на участке (0; t_fin)

# Характеристики различных предметов системы
phi_0 = math.pi / 2
A1A = B1B = 0.75 # длина параллельных стержней
AB = 1 # Длина прикреплённого стержня
OC = 0.125 # Расстояние от шарнира O до центра масс диска
r = 0,25 # Радиус диска

# Случайно-подобранные функции
phi = np.cos(2 * t) - np.sin(2 * t) - 3.15
tetta = phi_0 + 3 * np.cos(2 * t) - 3 * np.sin(2 * t)

# Координаты подвижных частей стержней
# Конец стержня A1A
A_X = A1A * np.sin(phi)
A_Y = A1A * np.cos(phi)
# Конец стержня B1B
B_X = B1B * np.sin(phi) + AB
B_Y = B1B * np.cos(phi)
# Центр масс диска
C_X = A_X + AB / 2 + OC * np.sin(tetta)
C_Y = A_Y + OC * np.cos(tetta)

fig = plt.figure(figsize=[10, 9]) # Создаём окно для отрисовки

ax = fig.add_subplot(1, 1, 1) # Добавляем ячейку (окно) для отрисовки
ax.axis('equal') # Делаем единичные отрезки осей равными
ax.set(xlim=[-3, 4], ylim=[-1.5, 1.5]) # Пределы по осям

# Координаты поверхности, на которой закрепелны стержни
X_Ground = [-0.5, 0, 1.5]
Y_Ground = [0.1, 0.1, 0.1]

# plot() рисует прямые линии, соединяя указанные точки
ax.plot(X_Ground, Y_Ground, color='black', linewidth=3) # Отрисовка поверхности

# Опоры, на которых вращаются стержни
Drawed_Q1A1 = ax.plot([-0.1, 0], [0.1, 0], color='black')
Drawed_Q2A1 = ax.plot([0.1, 0], [0.1, 0], color='black')
Drawed_E1A1 = ax.plot([AB - 0.1, AB], [0.1, 0], color='black')
Drawed_E2A1 = ax.plot([AB + 0.1, AB], [0.1, 0], color='black')

# Параметры, куда помещаем объект отрисовки чего-либо в начальный момент времени
# Стержени
Drawed_A1A = ax.plot([0, A_X[0]], [0, A_Y[0]], color='green')[0]
Drawed_B1B = ax.plot([1, B_X[0]], [0, B_Y[0]], color='green')[0]
Drawed_AB = ax.plot([A_X[0], B_X[0]], [A_Y[0], B_Y[0]], color='blue')[0]
# Радиус-вектор
Drawed_OC = ax.plot([A_X[0] + AB / 2, C_X[0]], [A_Y[0], C_Y[0]], color='black')[0]

# Отрисовываем точки для стержней
Point_A1 = ax.plot(0, 0, marker='o', color='black')
Point_B1 = ax.plot(1, 0, marker='o', color='black')
Point_A = ax.plot(A_X[0], A_Y[0], marker='o', color='green')[0]
Point_B = ax.plot(B_X[0], B_Y[0], marker='o', color='green')[0]
Point_O = ax.plot([A_X[0] + AB / 2], A_Y[0], marker='o', color='black')[0]
Point_C = ax.plot(C_X[0], C_Y[0], marker='o', markersize=70, markerfacecolor='none', markeredgewidth=2, color='red')[0]

# Обновление текущего кадра
def anima(i):
    # Изменяем нулевые элементы на i-ые:
    Drawed_A1A.set_data([0, A_X[i]], [0, A_Y[i]])
    Drawed_B1B.set_data([1, B_X[i]], [0, B_Y[i]])
    Drawed_AB.set_data([A_X[i], B_X[i]], [A_Y[i], B_Y[i]])
    Point_A.set_data(A_X[i], A_Y[i])
    Point_B.set_data(B_X[i], B_Y[i])
    Point_O.set_data([A_X[i] + AB / 2], A_Y[i])
    Point_C.set_data(C_X[i], C_Y[i])
    Drawed_OC.set_data([A_X[i] + AB / 2, C_X[i]], [A_Y[i], C_Y[i]])

    return [Point_A, Point_B, Point_O, Point_C, Drawed_A1A, Drawed_B1B, Drawed_AB, Drawed_OC]

anim = FuncAnimation(fig, anima, frames=len(t), interval=10, repeat=False) # Создаём анимацию

plt.show() # Отрисовываем