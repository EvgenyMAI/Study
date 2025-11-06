#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#define CSC(call)  									                \
do {											                    \
	cudaError_t res = call;							                \
	if (res != cudaSuccess) {							            \
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);								                    \
	}										                        \
} while(0)

const double EPS_ZERO = 1e-7;

__global__ void swap_rows_kernel(double *A, int n, int r1, int r2) {
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    int gridW = gridDim.x * blockDim.x;
    int gridH = gridDim.y * blockDim.y;
    int totalThreads = gridW * gridH;
    int gid = gy * gridW + gx;
    for (int t = gid; t < n; t += totalThreads) {
        int col = t;
        double tmp = A[(size_t)col * n + r1];
        A[(size_t)col * n + r1] = A[(size_t)col * n + r2];
        A[(size_t)col * n + r2] = tmp;
    }
}

__global__ void compute_factors_kernel(const double *A, double *f, int n, int k, double pivot) {
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    int gridW = gridDim.x * blockDim.x;
    int gridH = gridDim.y * blockDim.y;

    int rows = n - (k + 1);
    if (rows <= 0) {
        return;
    }

    for (int row_off = gx; row_off < rows; row_off += gridW) {
        int i = k + 1 + row_off;
        f[row_off] = A[(size_t)k * n + i] / pivot;
    }
}

__global__ void eliminate_columns_kernel(double *A, const double *f, int n, int k) {
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    int gridW = gridDim.x * blockDim.x;
    int gridH = gridDim.y * blockDim.y;

    int cols_to_do = n - (k + 1);
    int rows_to_do = n - (k + 1);
    if (cols_to_do <= 0 || rows_to_do <= 0) {
        return;
    }
    
    for (int row_off = gx; row_off < rows_to_do; row_off += gridW) {
        int i = k + 1 + row_off;
        for (int col_off = gy; col_off < cols_to_do; col_off += gridH) {
            int col = k + 1 + col_off;
            double Acolk = A[(size_t)col * n + k];
            double val = A[(size_t)col * n + i];
            val -= f[row_off] * Acolk;
            A[(size_t)col * n + i] = val;
        }
    }
}

struct AbsLess {
    __host__ __device__ bool operator()(const double &a, const double &b) const {
        return fabs(a) < fabs(b);
    }
};

int main() {
    int n;
    if (scanf("%d", &n) != 1 || n <= 0) {
        printf("0.0000000000e+00\n");
        return 0;
    }

    size_t total = (size_t)n * (size_t)n;
    double *hA = (double*)malloc(sizeof(double) * total);
    double v;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            scanf("%lf", &v);
            hA[(size_t)j * n + i] = v;
        }
    }

    double *dA, *dF;
    CSC(cudaMalloc(&dA, sizeof(double) * total));
    CSC(cudaMemcpy(dA, hA, sizeof(double) * total, cudaMemcpyHostToDevice));
    CSC(cudaMalloc(&dF, sizeof(double) * n));

    long long swaps = 0;
    double det = 1.0;

    dim3 fixedGrid(32, 32);
    dim3 fixedBlock(32, 8);

    for (int k = 0; k < n; ++k) {
        double *col_ptr = dA + (size_t)k * n + k;
        int len = n - k;
        thrust::device_ptr<double> dp = thrust::device_pointer_cast(col_ptr);
        thrust::device_ptr<double> max_it = thrust::max_element(dp, dp + len, AbsLess());
        int rel = (int)(max_it - dp);
        int pivot_row = k + rel;
        double pivot;
        CSC(cudaMemcpy(&pivot, dA + (size_t)k * n + pivot_row, sizeof(double), cudaMemcpyDeviceToHost));

        if (fabs(pivot) < EPS_ZERO) {
            printf("%.10e\n", 0.0);
            CSC(cudaFree(dF)); CSC(cudaFree(dA)); free(hA);
            return 0;
        }

        if (pivot_row != k) {
            swap_rows_kernel<<<fixedGrid, fixedBlock>>>(dA, n, k, pivot_row);
            CSC(cudaGetLastError());
            ++swaps;
            CSC(cudaMemcpy(&pivot, dA + (size_t)k * n + k, sizeof(double), cudaMemcpyDeviceToHost));
        }

        det *= pivot;
        if (k == n - 1) {
            break;
        }

        int rows = n - (k + 1);
        if (rows > 0) {
            compute_factors_kernel<<<fixedGrid, fixedBlock>>>(dA, dF, n, k, pivot);
            CSC(cudaGetLastError());
        }

        int cols = n - (k + 1);
        if (cols > 0 && rows > 0) {
            eliminate_columns_kernel<<<fixedGrid, fixedBlock>>>(dA, dF, n, k);
            CSC(cudaGetLastError());
        }
    }

    if (swaps % 2 != 0) {
        det = -det;
    }
    printf("%.10e\n", det);

    CSC(cudaFree(dF));
    CSC(cudaFree(dA));
    free(hA);
    return 0;
}