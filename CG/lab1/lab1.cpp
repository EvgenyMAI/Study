#include <GL/freeglut.h>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>

std::vector<std::pair<float, float>> controlPoints = {{50, 50}, {350, 50}, {350, 650}, {650, 650}};
int selectedPoint = -1;
float paramT = 0.0f;
bool isPlay = false;

// Вычисление вспомогательных точек для текущего уровня
// Возвращает новый набор точек, каждая из которых — результат линейной интерполяции между двумя соседними точками из входного вектора.
std::vector<std::pair<float, float>> calculateIntermediatePoints(const std::vector<std::pair<float, float>> &points, float t) {
    std::vector<std::pair<float, float>> result;
    for (size_t i = 0; i < points.size() - 1; ++i) {
        float newX = (1 - t) * points[i].first + t * points[i + 1].first;
        float newY = (1 - t) * points[i].second + t * points[i + 1].second;
        result.emplace_back(newX, newY);
    }
    return result;
}

// Отрисовка вспомогательных линий
void drawHelperLines(const std::vector<std::pair<float, float>> &points, float t, int depth = 0) {
    if (points.size() < 2) return;
    
    // Уникальный цвет для каждого уровня
    float colors[3][3] = {
        {0.0f, 1.0f, 0.0f},  // Зеленый
        {1.0f, 0.5f, 0.0f},  // Оранжевый
        {0.0f, 0.5f, 1.0f},  // Синий
    };
    glColor3f(colors[depth % 3][0], colors[depth % 3][1], colors[depth % 3][2]);

    // Линия соединяет все точки текущего уровня интерполяции
    glBegin(GL_LINE_STRIP); // Отрисовка линии, проходящей через все заданные вершины
    for (const auto &[x, y] : points) {
        glVertex2f(x, y);
    }
    glEnd();

    // Рекурсивная отрисовка линий следующего уровня
    auto nextLevel = calculateIntermediatePoints(points, t);
    drawHelperLines(nextLevel, t, depth + 1);

    // Отображение конечной точки
    if (nextLevel.size() == 1) {
        glColor3f(1.0f, 0.0f, 0.0f);  // Краснаый цвет
        glBegin(GL_POLYGON);
        // Окружность с радиусом 4
        for (float angle = 0; angle < 2 * M_PI; angle += 0.1f) {
            float x = nextLevel[0].first + 4 * cos(angle);
            float y = nextLevel[0].second + 4 * sin(angle);
            glVertex2f(x, y);
        }
        glEnd();
    }
}

// Отрисовка сетки
void drawGrid() {
    glColor3f(0.9f, 0.9f, 0.9f);
    for (int i = 0; i <= 700; i += 20) {
        glBegin(GL_LINES);
        glVertex2f(i, 0);
        glVertex2f(i, 700);
        glVertex2f(0, i);
        glVertex2f(700, i);
        glEnd();
    }
}

// Отрисовка контрольных точек
void drawControlPoints() {
    for (size_t i = 0; i < controlPoints.size(); ++i) {
        const auto &[x, y] = controlPoints[i];
        
        // Отображение контрольной точки
        glBegin(GL_POLYGON);
        glColor3f(1.0f, 1.0f, 0.0f);  // Желтый цвет
        for (float angle = 0; angle < 2 * M_PI; angle += 0.1f) {
            float dx = x + 6 * cos(angle);
            float dy = y + 6 * sin(angle);
            glVertex2f(dx, dy);
        }
        glEnd();

        // Отображение номера точки рядом с ней
        glColor3f(0.0f, 0.0f, 0.0f);  // Черный цвет для текста
        std::stringstream ss;
        ss << (i + 1);  // Номер точки
        glRasterPos2f(x + 10, y + 10);  // Позиция для текста
        std::string text = ss.str();
        for (const char &c : text) {
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, c);
        }
    }
}

// Отрисовка кривой Безье
void drawBezierCurve() {
    glColor3f(1.0f, 0.0f, 0.0f);  // Красная кривая
    glBegin(GL_LINE_STRIP); // Отрисовка линии, проходящей через все указанные вершины
    for (float t = 0; t <= 1; t += 0.01f) {
        auto intermediatePoints = controlPoints;
        // Линейная интерполяция точек до момента, пока не останется 1 точка - точка прямой
        while (intermediatePoints.size() > 1) {
            intermediatePoints = calculateIntermediatePoints(intermediatePoints, t);
        }
        glVertex2f(intermediatePoints[0].first, intermediatePoints[0].second); // Добавление вершины в список вершин линии
    }
    glEnd();
}

// Отрисовка кнопки состояния и значения t
void drawUI() {
    glColor3f(0.0f, 0.0f, 0.0f);
    glRasterPos2f(10, 680);
    const char *state = isPlay ? "Play" : "Pause";
    for (const char *c = state; *c != '\0'; ++c) glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *c);

    // Отображение значения t
    std::stringstream ss;
    ss << "t: " << std::fixed << std::setprecision(3) << paramT;
    glRasterPos2f(10, 655);
    std::string tValue = ss.str();
    for (const char &c : tValue) glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, c);
}

// Основная функция отрисовки
void display() {
    glClear(GL_COLOR_BUFFER_BIT);
    drawGrid();
    drawBezierCurve();
    drawHelperLines(controlPoints, paramT);
    drawControlPoints();
    drawUI();
    glutSwapBuffers(); // Переключение переднего и заднего буферов для двойной буферизации
}

// Таймер для анимации
void timer(int value) {
    if (isPlay) {
        paramT += 0.003f;
        if (paramT >= 1.0f) {
            paramT = 0.0f;
        }
    }
    glutPostRedisplay(); // Вызывает функцию отрисовки display() в ближайшем цикле обработки событий
    glutTimerFunc(16, timer, 0); // Частота вызова функции timer() = 16 мс ~ 60 FPS
}

// Обработчик нажатий мыши
void mouse(int button, int state, int x, int y) {
    if (state == GLUT_DOWN) {
        //Вычисляем расстояние между каждой точкой и текущей позицией мыши
        for (size_t i = 0; i < controlPoints.size(); ++i) {
            float dx = x - controlPoints[i].first;
            float dy = (700 - y) - controlPoints[i].second;
            if (sqrt(dx * dx + dy * dy) < 10) {
                selectedPoint = i;
                return;
            }
        }
    } else if (state == GLUT_UP) {
        selectedPoint = -1;
    }
}

// Обработчик перемещения мыши
void motion(int x, int y) {
    if (selectedPoint != -1) {
        controlPoints[selectedPoint] = {x, 700 - y};
        glutPostRedisplay();
    }
}

// Обработчик клавиш
void keyboard(unsigned char key, int, int) {
    if (key == ' ') isPlay = !isPlay;
}

// Инициализация OpenGL
void init() {
    glMatrixMode(GL_PROJECTION); // Переключает текущий режим матрицы на проекционную матрицу
    glLoadIdentity(); // Сбрасывает текущую матрицу в единичную матрицу
    gluOrtho2D(0, 700, 0, 700); // Устанавливает ортографическую проекцию для двухмерной отрисовки
    glClearColor(0.95f, 0.95f, 0.95f, 1.0f); // Устанавливает цвет, которым будет заполнен экран при очистке
}

int main(int argc, char **argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(700, 700);
    glutCreateWindow("Кубическая кривая Безье");
    init();
    glutDisplayFunc(display); // Регистрирует функцию display() в качестве функции обратного вызова для перерисовки окна
    glutTimerFunc(25, timer, 0);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutKeyboardFunc(keyboard);
    glutMainLoop(); // Запускает главный цикл обработки событий GLUT
    return 0;
}