#include <GL/freeglut.h>
#include <GL/gl.h>
#include <cstdio>
#include <cmath>

float pyramidScale = 1.0f;
float cameraDistance = 5.0f;
float cameraAngleX = 0.0f;
float cameraAngleY = 0.0f;

char lastKey = '\0';
int lastMouseX = -1, lastMouseY = -1;

// Вывод текста на экран
void renderText(float x, float y, const char* text) {
    glRasterPos2f(x, y);
    for (const char* c = text; *c != '\0'; c++) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *c);
    }
}

// Построение пирамиды
void drawPyramid() {
    glPushMatrix(); // Матрица трансформаций
    glScalef(pyramidScale, pyramidScale, pyramidScale);

    glColor3f(0.0f, 0.5f, 1.0f); // Голубой цвет для пирамиды
    glBegin(GL_TRIANGLES);

    // Вершина пирамиды
    float apex[3] = {0.0f, 1.0f, 0.0f};

    // Основание (четыре вершины)
    float base[4][3] = {
        {-1.0f, 0.0f, -1.0f},
        { 1.0f, 0.0f, -1.0f},
        { 1.0f, 0.0f,  1.0f},
        {-1.0f, 0.0f,  1.0f}
    };

    // Треугольные грани
    for (int i = 0; i < 4; i++) {
        glVertex3fv(apex);
        glVertex3fv(base[i]);
        glVertex3fv(base[(i + 1) % 4]);
    }
    glEnd();

    // Основание пирамиды
    glColor3f(0.0f, 0.3f, 0.7f); // Тёмно-голубой цвет
    glBegin(GL_QUADS);
    for (int i = 0; i < 4; i++) {
        glVertex3fv(base[i]);
    }
    glEnd();

    glPopMatrix();
}

// Обновление камеры
void updateCamera() {
    glLoadIdentity(); // Сбрасывает текущую матрицу
    gluLookAt( // Определяет положение камеры и направление взгляда
        cameraDistance * cos(cameraAngleY) * cos(cameraAngleX), // X
        cameraDistance * sin(cameraAngleY),                     // Y
        cameraDistance * cos(cameraAngleY) * sin(cameraAngleX), // Z
        0.0f, 0.0f, 0.0f,  // Центр сцены
        0.0f, 1.0f, 0.0f   // Вектор вверх
    );
}

// Отрисовка
void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    // Отрисовка 3D сцены
    updateCamera();
    drawPyramid();

    // Переход в 2D-режим для отрисовки текста
    glMatrixMode(GL_PROJECTION); // Переключаем матрицу на проекционную
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, 800, 0, 600); // Устанавливаем ортографическую проекцию
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity(); // Сбрасывает текущую матрицу в единичную матрицу

    // Отображение информации
    glColor3f(0.0f, 0.0f, 0.0f); // Черный цвет текста
    renderText(10, 40, "W/S: camera zoom in/out");
    renderText(10, 20, "A/D: increase/decrease pyramid scale");

    char info[128];
    sprintf(info, "Camera Distance: %.2f", cameraDistance);
    renderText(10, 580, info);
    sprintf(info, "Pyramid Scale: %.2f", pyramidScale);
    renderText(10, 560, info);

    if (lastKey != '\0') {
        sprintf(info, "Last Key Pressed: %c", lastKey);
        renderText(10, 540, info);
    }

    // Возвращение в 3D-режим
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);

    glutSwapBuffers();
}

void initOpenGL() {
    glEnable(GL_DEPTH_TEST);  // Включение теста глубины для корректного отображения 3D объектов
    glClearColor(0.95f, 0.95f, 0.95f, 1.0f);  // Светлый фон
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0f, 1.0f, 0.1f, 100.0f);  // Перспективная проекция
    glMatrixMode(GL_MODELVIEW);
}

// Обработчик клавиш
void keyboard(unsigned char key, int x, int y) {
    lastKey = key;
    if (key == 'w') {
        cameraDistance -= 0.1f;
        if (cameraDistance < 1.0f) cameraDistance = 1.0f;  // Минимальное расстояние
    } else if (key == 's') {
        cameraDistance += 0.1f;
    } else if (key == 'a') {
        pyramidScale -= 0.1f;
        if (pyramidScale < 0.5f) pyramidScale = 0.5f;  // Минимальный масштаб
    } else if (key == 'd') {
        pyramidScale += 0.1f;
        if (pyramidScale > 3.0f) pyramidScale = 3.0f;  // Максимальный масштаб
    }
    glutPostRedisplay();
}

// Обработчик движения мыши
void motion(int x, int y) {
    // Инициализация при первом движении мыши
    if (lastMouseX == -1 || lastMouseY == -1) {
        lastMouseX = x;
        lastMouseY = y;
    }

    // Вычисляем разницу между текущими и предыдущими координатами
    int deltaX = x - lastMouseX;
    int deltaY = y - lastMouseY;

    // Обновляем углы камеры
    cameraAngleX += deltaX * 0.01f;
    cameraAngleY += deltaY * 0.01f;

    // Сохраняем текущие координаты мыши
    lastMouseX = x;
    lastMouseY = y;

    glutPostRedisplay();
}

// Обработчик кнопок мыши (сбрасывает начальные значения координат мыши при новом клике, чтобы избежать резкого скачка при движении мыши)
void mouse(int button, int state, int x, int y) {
    if (state == GLUT_DOWN) {
        lastMouseX = x;
        lastMouseY = y;
    }
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(800, 600);
    glutCreateWindow("3D Pyramid");
    
    initOpenGL();

    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMotionFunc(motion);
    glutMouseFunc(mouse);

    glutMainLoop();
    return 0;
}