#include <stdio.h>
#include <math.h>

int main() {
    float a, b, c;
    
    if (scanf("%f %f %f", &a, &b, &c) != 3) {
        printf("incorrect");
        return 0;
    }
    if (a == 0 && b == 0 && c == 0) {
        printf("any");
        return 0;
    }
    if (a == 0 && b == 0 && c != 0) {
        printf("incorrect");
        return 0;
    }
    if (a == 0) {
        float x = -c / b;
        printf("%.6f", x);
        return 0;
    }
    
    float D = b * b - 4 * a * c;
    if (D > 0) {
        float sqrtD = sqrtf(D);
        float x1 = (-b + sqrtD) / (2 * a);
        float x2 = (-b - sqrtD) / (2 * a);
        printf("%.6f %.6f", x1, x2);
    } else if (D == 0) {
        float x = -b / (2 * a);
        printf("%.6f", x);
    } else {
        printf("imaginary");
    }
    
    return 0;
}