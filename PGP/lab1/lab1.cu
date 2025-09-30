#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void kernel(double *a, double *b, double *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;
    while (idx < n) {
        out[idx] = fmin(a[idx], b[idx]);
        idx += offset;
    }
}

int main() {
    int n;
    if (scanf("%d", &n) != 1) {
      return 0;
    }

    double *a = (double *)malloc(sizeof(double) * n);
    for (int i = 0; i < n; ++i)
        scanf("%lf", &a[i]);

    double *b = (double *)malloc(sizeof(double) * n);
    for (int i = 0; i < n; ++i)
        scanf("%lf", &b[i]);

    double *dev_a, *dev_b, *dev_out;
    cudaMalloc(&dev_a, sizeof(double) * n);
    cudaMalloc(&dev_b, sizeof(double) * n);
    cudaMalloc(&dev_out, sizeof(double) * n);

    cudaMemcpy(dev_a, a, sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, sizeof(double) * n, cudaMemcpyHostToDevice);

    kernel<<<1024, 1024>>>(dev_a, dev_b, dev_out, n);

    double *out = (double *)malloc(sizeof(double) * n);
    cudaMemcpy(out, dev_out, sizeof(double) * n, cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) {
        printf("%.10e", out[i]);
        if (i + 1 < n) {
          printf(" ");
        }
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_out);
    free(a);
    free(b);
    free(out);

    return 0;
}
