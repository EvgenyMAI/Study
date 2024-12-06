#include "lib.h"

int GCF(int A, int B) {
    if (B == 0)
        return A;
	else
        return GCF(B, A % B);
}

float Square(float A, float B) {
    if (A <= 0 || B <= 0) {
        printf("Ошибка: стороны должны быть положительными числами.\n");
        return -1;
    }
    
    float area = A * B;
    return area;
}