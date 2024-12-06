#include "lib.h"

int GCF(int A, int B) {
    int gcf = 1;
    for (int i = 1; i <= A && i <= B; ++i) {
        if (A % i == 0 && B % i == 0)
            gcf = i;
    }

    return gcf;
}

float Square(float A, float B) {
    if (A <= 0 || B <= 0) {
        printf("Ошибка: стороны должны быть положительными числами.\n");
        return -1;
    }
    
    float area = 0.5 * A * B;
    return area;
}