#include <iostream>
#include <vector>
#include <cmath>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

struct Ray {
    glm::vec3 origin; // Начальная точка луча
    glm::vec3 direction;

    Ray(const glm::vec3& o, const glm::vec3& d) : origin(o), direction(d) {}
};

struct Sphere {
    glm::vec3 center;
    float radius;
    glm::vec3 color;
    float reflectivity;

    Sphere(const glm::vec3& c, float r, const glm::vec3& col, float refl = 0.5f) : center(c), radius(r), color(col), reflectivity(refl) {}

    // Вычисляет пересечение луча с поверхностью сферы
    bool intersect(const Ray& ray, float& t) const {
        glm::vec3 oc = ray.origin - center;
        float a = glm::dot(ray.direction, ray.direction);
        float b = 2.0f * glm::dot(oc, ray.direction);
        float c = glm::dot(oc, oc) - radius * radius;
        float discriminant = b * b - 4.0f * a * c;
        
        if (discriminant > 0) {
            t = (-b - sqrt(discriminant)) / (2.0f * a);
            return t > 0;
        }
        return false;
    }

    // Вычисляет нормаль в точке пересечения
    glm::vec3 getNormal(const glm::vec3& point) const {
        return glm::normalize(point - center);
    }
};

struct Plane {
    glm::vec3 point;
    glm::vec3 normal;
    glm::vec3 color;
    float reflectivity;

    Plane(const glm::vec3& p, const glm::vec3& n, const glm::vec3& col, float refl = 0.5f) : point(p), normal(n), color(col), reflectivity(refl) {}

    bool intersect(const Ray& ray, float& t) const {
        float denom = glm::dot(normal, ray.direction);
        if (fabs(denom) > 1e-6) {
            glm::vec3 p0l0 = point - ray.origin;
            t = glm::dot(p0l0, normal) / denom;
            return t >= 0;
        }
        return false;
    }

    glm::vec3 getNormal() const {
        return normal;
    }
};

// Объявление функции trace
glm::vec3 trace(const Ray& ray, const std::vector<Sphere>& spheres, const std::vector<Plane>& planes, 
                const glm::vec3& lightPos, const glm::vec3& lightColor, int depth);

// Вычисление освещения
glm::vec3 computeLighting(const glm::vec3& hitPoint, const glm::vec3& normal, const glm::vec3& viewDir,
                          const glm::vec3& lightPos, const glm::vec3& lightColor, const std::vector<Sphere>& spheres, 
                          const std::vector<Plane>& planes, int depth) {
    glm::vec3 lightDir = glm::normalize(lightPos - hitPoint);
    glm::vec3 halfDir = glm::normalize(lightDir + viewDir);

    // Освещение (Phong модель)
    float diffuse = std::max(glm::dot(normal, lightDir), 0.0f);
    float specular = std::pow(std::max(glm::dot(normal, halfDir), 0.0f), 32.0f);

    // Отражения
    glm::vec3 reflectionColor(0.0f);
    if (depth < 4) {
        glm::vec3 reflectDir = glm::normalize(glm::reflect(-lightDir, normal));
        Ray reflectRay(hitPoint + normal * 0.001f, reflectDir);
        reflectionColor = trace(reflectRay, spheres, planes, lightPos, lightColor, depth + 1) * 0.5f; // Снижение яркости отражений
    }

    // Косвенное освещение
    glm::vec3 indirectColor(0.0f);
    if (depth < 3) {
        glm::vec3 randomDirection = glm::normalize(glm::vec3(
            (rand() % 2000 - 1000) / 1000.0f,
            (rand() % 2000 - 1000) / 1000.0f,
            (rand() % 2000 - 1000) / 1000.0f
        ));
        Ray indirectRay(hitPoint + normal * 0.001f, randomDirection);
        indirectColor = trace(indirectRay, spheres, planes, lightPos, lightColor, depth + 1) * 0.3f;
    }

    // Мягкие тени
    float shadow = 0.0f;
    int numShadowRays = 6;  // Количество лучей для мягкой тени
    for (int i = 0; i < numShadowRays; ++i) {
        glm::vec3 jitteredLightPos = lightPos + glm::vec3(
            (rand() % 100 - 50) / 50.0f, 
            (rand() % 100 - 50) / 50.0f, 
            (rand() % 100 - 50) / 50.0f
        );
        Ray shadowRay(hitPoint + normal * 0.001f, glm::normalize(jitteredLightPos - hitPoint));
        bool inShadow = false;

        // Проверяем пересечения с объектами
        for (const auto& sphere : spheres) {
            float tNear;
            if (sphere.intersect(shadowRay, tNear)) {
                inShadow = true;
                break;
            }
        }
        for (const auto& plane : planes) {
            float tNear;
            if (plane.intersect(shadowRay, tNear)) {
                inShadow = true;
                break;
            }
        }

        shadow += inShadow ? 1.0f : 0.0f;
    }
    shadow /= float(numShadowRays);

    // Итоговый цвет
    glm::vec3 color = (lightColor * diffuse * (1.0f - shadow) + reflectionColor * specular + indirectColor) * 0.6f;
    return color;
}

// Трассировка лучей с учетом отражений и глобального освещения
glm::vec3 trace(const Ray& ray, const std::vector<Sphere>& spheres, const std::vector<Plane>& planes, 
                const glm::vec3& lightPos, const glm::vec3& lightColor, int depth) {
    float tNear = INFINITY;
    const Sphere* hitSphere = nullptr;
    const Plane* hitPlane = nullptr;

    // Проверка пересечений с объектами
    for (const auto& sphere : spheres) {
        float t = 0;
        if (sphere.intersect(ray, t) && t < tNear) {
            tNear = t;
            hitSphere = &sphere;
        }
    }
    for (const auto& plane : planes) {
        float t = 0;
        if (plane.intersect(ray, t) && t < tNear) {
            tNear = t;
            hitPlane = &plane;
        }
    }

    // Вычисление освещения для пересеченной точки
    if (hitSphere) {
        glm::vec3 hitPoint = ray.origin + ray.direction * tNear;
        glm::vec3 normal = hitSphere->getNormal(hitPoint);
        glm::vec3 viewDir = -ray.direction;

        return computeLighting(hitPoint, normal, viewDir, lightPos, lightColor, spheres, planes, depth);
    } else if (hitPlane) {
        glm::vec3 hitPoint = ray.origin + ray.direction * tNear;
        glm::vec3 normal = hitPlane->getNormal();
        glm::vec3 viewDir = -ray.direction;

        return computeLighting(hitPoint, normal, viewDir, lightPos, lightColor, spheres, planes, depth);
    }

    // Возвращаем черный цвет, если нет пересечений
    return glm::vec3(0.0f);
}

// Основная функция рендеринга с сглаживанием (антиалиасинг)
void renderScene() {
    std::vector<Sphere> spheres = {
        Sphere(glm::vec3(0.0f, 0.0f, -5.0f), 1.0f, glm::vec3(0.8f, 0.1f, 0.1f)),
        Sphere(glm::vec3(2.0f, 0.0f, -6.0f), 1.0f, glm::vec3(0.1f, 0.8f, 0.1f))
    };

    std::vector<Plane> planes = {
        Plane(glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.8f, 0.8f, 0.8f))
    };

    glm::vec3 lightPos(-4.0f, 5.0f, -3.0f);
    glm::vec3 lightColor(1.0f, 1.0f, 1.0f);

    int samples = 4; // Количество сэмплов для сглаживания

    // Отправка лучей для трассировки
    for (int y = -400; y < 400; ++y) {
        for (int x = -400; x < 400; ++x) {
            glm::vec3 accumulatedColor(0.0f);

            // Сглаживание с несколькими сэмплами
            for (int sy = 0; sy < samples; ++sy) {
                for (int sx = 0; sx < samples; ++sx) {
                    // Каждый пиксель делится на несколько частей для сглаживания
                    float xNDC = (float)(x + (float(sx) / samples)) / 400.0f; // нормализованные координаты пикселя с учетом сдвига сэмпла
                    float yNDC = (float)(y + (float(sy) / samples)) / 400.0f;

                    // Луч из камеры (вывод на экран)
                    glm::vec3 rayDir = glm::normalize(glm::vec3(xNDC, yNDC, -1.0f));
                    Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), rayDir);

                    accumulatedColor += trace(ray, spheres, planes, lightPos, lightColor, 0);
                }
            }

            // Средний цвет в текущем пикселе для сглаживания
            accumulatedColor /= float(samples * samples);

            // Отображение пикселя
            glColor3f(accumulatedColor.r, accumulatedColor.g, accumulatedColor.b);
            glBegin(GL_POINTS);
            glVertex2f(x / 400.0f, y / 400.0f);
            glEnd();
        }
    }
    glFlush();
}

int main() {
    // Инициализация OpenGL
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    GLFWwindow* window = glfwCreateWindow(800, 800, "Ray Tracing", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glewInit();

    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        renderScene();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}