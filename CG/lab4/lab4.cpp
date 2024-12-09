#include <GL/glew.h>
#include <GL/freeglut.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <vector>

GLuint shaderProgram;
// VAO (Vertex Array Object) хранит настройки привязки буферов.
// VBO (Vertex Buffer Object) хранит вершины.
// EBO (Element Buffer Object) хранит индексы.
GLuint VAO, VBO, EBO;
float cubeAngleX = 0.0f, cubeAngleY = 0.0f;
int lastMouseX = -1, lastMouseY = -1;
bool useFlatShading = true;

// Вершинный шейдер
// R"(...)" — это "raw string literal"
const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

out vec3 FragPos;
out vec3 Normal;
out vec3 LightDir;
out vec3 GouraudColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat3 normalMatrix;  // Матрица нормалей
uniform vec3 lightPos;
uniform vec3 lightColor;
uniform vec3 objectColor;
uniform bool useFlatShading;

void main()
{
    // Мировые координаты
    FragPos = vec3(model * vec4(aPos, 1.0));
    
    // Нормаль в мировых координатах
    Normal = normalMatrix * aNormal;

    // Направление света
    LightDir = normalize(lightPos - FragPos);

    if (!useFlatShading) {
        // Gouraud Shading: рассчитываем цвет на вершинном шейдере
        float diff = max(dot(normalize(Normal), LightDir), 0.0);
        GouraudColor = diff * lightColor * objectColor;
    }

    // Преобразование вершин
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
)";

// Фрагментный шейдер
const char* fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;
in vec3 LightDir;
in vec3 GouraudColor;

uniform vec3 lightColor;
uniform vec3 objectColor;
uniform bool useFlatShading;
uniform vec3 ambientLight;

void main()
{
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(LightDir);

    if (useFlatShading) {
        // Flat Shading: расчет цвета на основе нормали (делается в фрагментном шейдере)
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 flatColor = diff * lightColor * objectColor;
        FragColor = vec4(flatColor + ambientLight * objectColor, 1.0);  // Учитываем амбиентное освещение
    } else {
        // Gouraud Shading: используем уже интерполированный цвет
        FragColor = vec4(GouraudColor + ambientLight * objectColor, 1.0);  // Учитываем амбиентное освещение
    }
}
)";

// Функция для компиляции шейдеров
GLuint CompileShader(const char* shaderSource, GLenum shaderType) {
    GLuint shader = glCreateShader(shaderType);
    glShaderSource(shader, 1, &shaderSource, NULL);
    glCompileShader(shader);

    // Проверка на ошибки компиляции
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cerr << "Shader Compilation Error: " << infoLog << std::endl;
    }
    return shader;
}

// Функция для компиляции шейдерной программы
GLuint CreateShaderProgram() {
    GLuint vertexShader = CompileShader(vertexShaderSource, GL_VERTEX_SHADER);
    GLuint fragmentShader = CompileShader(fragmentShaderSource, GL_FRAGMENT_SHADER);
    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    // Проверка на ошибки линковки программы
    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, NULL, infoLog);
        std::cerr << "Program Linking Error: " << infoLog << std::endl;
    }

    // Очистка шейдеров
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return program;
}

// Функция для инициализации данных куба 
void InitCube() {
    GLfloat vertices[] = {
        // Позиции            // Нормали
        -0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
         0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
        -0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,

        -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
         0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
         0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
        -0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,

        -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,
        -0.5f,  0.5f, -0.5f, -1.0f,  0.0f,  0.0f,
        -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,
        -0.5f, -0.5f,  0.5f, -1.0f,  0.0f,  0.0f,

         0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,
         0.5f,  0.5f, -0.5f,  1.0f,  0.0f,  0.0f,
         0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,
         0.5f, -0.5f,  0.5f,  1.0f,  0.0f,  0.0f,

        -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,
         0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,
         0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,

        -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,
         0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,
        -0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f
    };

    // Индексы, которые определяют, какие вершины образуют треугольники для каждой из 6 сторон куба
    GLuint indices[] = {
        0, 1, 2, 2, 3, 0, // Перед
        4, 5, 6, 6, 7, 4, // Зад
        8, 9, 10, 10, 11, 8, // Лево
        12, 13, 14, 14, 15, 12, // Право
        16, 17, 18, 18, 19, 16, // Низ
        20, 21, 22, 22, 23, 20  // Верх
    };

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    // Привязываются и настраиваются буферы
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    //Устанавливаются два атрибута для каждой вершины:
    // Атрибут 0 (позиции вершин): 3 компонента (x, y, z).
    // Атрибут 1 (нормали вершин): 3 компонента (nx, ny, nz).
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (GLvoid*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

// Функция для обработки событий клавиатуры
void key_callback(unsigned char key, int x, int y) {
    if (key == 'f') {
        useFlatShading = !useFlatShading;
    }
}

// Функция для обработки ввода мыши (вращение куба)
void MouseMotion(int x, int y) {
    if (lastMouseX == -1 || lastMouseY == -1) {
        lastMouseX = x;
        lastMouseY = y;
    }

    int deltaX = x - lastMouseX;
    int deltaY = y - lastMouseY;

    cubeAngleX += deltaY * 0.1f;
    cubeAngleY += deltaX * 0.1f;

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

// Функция для отрисовки сцены
void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(shaderProgram);

    // Матрицы (используются для преобразования 3D-объекта в 2D-окне)
    glm::mat4 model = glm::rotate(glm::mat4(1.0f), glm::radians(cubeAngleX), glm::vec3(1.0f, 0.0f, 0.0f));
    model = glm::rotate(model, glm::radians(cubeAngleY), glm::vec3(0.0f, 1.0f, 0.0f));

    glm::mat4 view = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -3.0f));
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);

    // Получение локаций uniform-переменных в шейдерах
    GLuint modelLoc = glGetUniformLocation(shaderProgram, "model");
    GLuint viewLoc = glGetUniformLocation(shaderProgram, "view");
    GLuint projectionLoc = glGetUniformLocation(shaderProgram, "projection");
    GLuint normalMatrixLoc = glGetUniformLocation(shaderProgram, "normalMatrix");
    GLuint lightPosLoc = glGetUniformLocation(shaderProgram, "lightPos");
    GLuint lightColorLoc = glGetUniformLocation(shaderProgram, "lightColor");
    GLuint objectColorLoc = glGetUniformLocation(shaderProgram, "objectColor");
    GLuint ambientLightLoc = glGetUniformLocation(shaderProgram, "ambientLight");
    GLuint useFlatShadingLoc = glGetUniformLocation(shaderProgram, "useFlatShading");

    // Передача матриц в шейдер
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));

    // Передача нормальной матрицы
    glUniformMatrix3fv(normalMatrixLoc, 1, GL_FALSE, glm::value_ptr(glm::transpose(glm::inverse(glm::mat3(model)))));

    // Параметры освещения
    glm::vec3 lightPos(1.0f, 1.0f, 1.0f);
    glm::vec3 lightColor(1.0f, 1.0f, 1.0f);
    glm::vec3 objectColor(0.6f, 0.6f, 0.6f);
    glm::vec3 ambientLight(0.2f, 0.2f, 0.2f);

    glUniform3fv(lightPosLoc, 1, glm::value_ptr(lightPos));
    glUniform3fv(lightColorLoc, 1, glm::value_ptr(lightColor));
    glUniform3fv(objectColorLoc, 1, glm::value_ptr(objectColor));
    glUniform3fv(ambientLightLoc, 1, glm::value_ptr(ambientLight));
    glUniform1i(useFlatShadingLoc, useFlatShading);

    // Отрисовка куба
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

    glutSwapBuffers();
}

// Функция инициализации OpenGL
void initOpenGL() {
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    shaderProgram = CreateShaderProgram();
    InitCube();
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(800, 600);
    glutCreateWindow("Flat and Gouraud Shading");

    glewInit();
    
    initOpenGL();
    
    glutDisplayFunc(display);
    glutIdleFunc(display);
    glutMotionFunc(MouseMotion);
    glutMouseFunc(mouse);
    glutKeyboardFunc(key_callback);

    glutMainLoop();
    return 0;
}