import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation

# Функция составления дифференциальных уравнений
def odesys(y, t, g, m1, m2, r, l, b, k):
    dy = np.zeros(4)
    dy[0] = y[2] # phi с точкой
    dy[1] = y[3] # thetta с точкой

    a11 = (m1+m2)*l
    a12 = m1*b*np.cos(y[1]-y[0])
    a21 = l*np.cos(y[1]-y[0])
    a22 = ((r**2)/(2*b))+b

    b1 = -(m1+m2)*g*np.sin(y[0])+m1*b*((y[3]**2)*np.sin(y[1]-y[0]))
    b2 = -g*np.sin(y[1])-(k*y[3])/(m1*b)-l*(y[2]**2)*np.sin(y[1]-y[0])

    dy[2] = (b1*a22 - b2*a12)/(a11*a22 - a12*a21)
    dy[3] = (b2*a11 - b1*a21)/(a11*a22 - a12*a21)

    return dy

Steps = 1001
t_fin = 20
t = np.linspace(0, t_fin, Steps) # Равномерно распределяем Steps элементов на участке (0; t_fin)

# Характеристики различных предметов системы
g = 9.81
m1 = 50 # Масса диска
m2 = 50 # Масса прикреплённого стержня
r = 0.25 # Радиус диска
l = 1 # Длины стержней
b = 0.125 # Расстояние от шарнира O до центра масс диска
k = 10 # Константа, пропорциональная угловой скорости

# Начальные положения
phi_0 = math.pi / 2
thetta_0 = math.pi / 2
dphi_0 = 0
dthetta_0 = 0
y0 = [phi_0, thetta_0, dphi_0, dthetta_0]

# Интегрируем систему, которую создаёт odeint
Y = odeint(odesys, y0, t, args=(g, m1, m2, r, l, b, k))

phi = Y[:,0]
thetta = Y[:,1]
# Для графиков реакции
dphi = Y[:,2]
dthetta = Y[:,3]
ddphi = [odesys(y, t, g, m1, m2, r, l, b, k)[2] for y,t in zip(Y, t)]
ddthetta = [odesys(y, t, g, m1, m2, r, l, b, k)[3] for y,t in zip(Y, t)]
Rx = -m1*(l*(ddphi*np.sin(phi)+dphi**2*np.cos(phi))+b*(ddthetta*np.sin(thetta)+dthetta**2*np.cos(thetta)))-m1*g
Ry = m1*(l*(ddphi*np.cos(phi)-dphi**2*np.sin(phi))+b*(ddthetta*np.cos(thetta)-dthetta**2*np.sin(thetta)))

# Создаём дополнительное окно для отриовки графиков зависимости обобщённых координат от времени
fig_for_graphs = plt.figure(figsize=[8, 7])
# phi(t)
ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 1)
ax_for_graphs.plot(t, phi, color='Blue')
ax_for_graphs.set_title("phi(t)")
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True) # Добавляем сетку
# thetta(t)
ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 3)
ax_for_graphs.plot(t, thetta, color='Red')
ax_for_graphs.set_title("thetta(t)")
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)
# Rx
ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 2)
ax_for_graphs.plot(t, Rx, color='Black')
ax_for_graphs.set_title("Rx")
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)
# Ry
ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 4)
ax_for_graphs.plot(t, Ry, color='Gray')
ax_for_graphs.set_title("Ry")
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)

# Координаты подвижных частей стержней
# Конец стержня l
A_X = l * np.sin(phi)
A_Y = - l * np.cos(phi)
# Конец стержня l
B_X = l * np.sin(phi) + l
B_Y = - l * np.cos(phi)
# Центр масс диска
C_X = A_X + l / 2 + b * np.sin(thetta)
C_Y = A_Y - b * np.cos(thetta)

fig = plt.figure(figsize=[8, 7]) # Создаём окно для отрисовки

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
Drawed_E1A1 = ax.plot([l - 0.1, l], [0.1, 0], color='black')
Drawed_E2A1 = ax.plot([l + 0.1, l], [0.1, 0], color='black')

# Параметры, куда помещаем объект отрисовки чего-либо в начальный момент времени
# Стержени
Drawed_A1A = ax.plot([0, A_X[0]], [0, A_Y[0]], color='green')[0]
Drawed_B1B = ax.plot([1, B_X[0]], [0, B_Y[0]], color='green')[0]
Drawed_AB = ax.plot([A_X[0], B_X[0]], [A_Y[0], B_Y[0]], color='blue')[0]
# Радиус-вектор
Drawed_OC = ax.plot([A_X[0] + l / 2, C_X[0]], [A_Y[0], C_Y[0]], color='black')[0]

# Отрисовываем точки для стержней
Point_A1 = ax.plot(0, 0, marker='o', color='black')
Point_B1 = ax.plot(1, 0, marker='o', color='black')
Point_A = ax.plot(A_X[0], A_Y[0], marker='o', color='green')[0]
Point_B = ax.plot(B_X[0], B_Y[0], marker='o', color='green')[0]
Point_O = ax.plot([A_X[0] + l / 2], A_Y[0], marker='o', color='black')[0]
Point_C = ax.plot(C_X[0], C_Y[0], marker='o', markersize=70, markerfacecolor='none', markeredgewidth=2, color='red')[0]

# Обновление текущего кадра
def anima(i):
    # Изменяем нулевые элементы на i-ые:
    Drawed_A1A.set_data([0, A_X[i]], [0, A_Y[i]])
    Drawed_B1B.set_data([1, B_X[i]], [0, B_Y[i]])
    Drawed_AB.set_data([A_X[i], B_X[i]], [A_Y[i], B_Y[i]])
    Point_A.set_data(A_X[i], A_Y[i])
    Point_B.set_data(B_X[i], B_Y[i])
    Point_O.set_data([A_X[i] + l / 2], A_Y[i])
    Point_C.set_data(C_X[i], C_Y[i])
    Drawed_OC.set_data([A_X[i] + l / 2, C_X[i]], [A_Y[i], C_Y[i]])

    return [Point_A, Point_B, Point_O, Point_C, Drawed_A1A, Drawed_B1B, Drawed_AB, Drawed_OC]

anim = FuncAnimation(fig, anima, frames=len(t), interval=10, repeat=False) # Создаём анимацию

plt.show() # Отрисовываем