#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <time.h>
#include <cuda_runtime.h>

#define MAX_CLASSES 32

#define CSC(call) \
do { \
    cudaError_t res = call; \
    if (res != cudaSuccess) { \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n", __FILE__, __LINE__, cudaGetErrorString(res)); \
        exit(0); \
    } \
} while(0)

typedef unsigned char uchar;

typedef struct {
    uchar x, y, z, w;
} uchar4_custom;

__constant__ float c_avg[MAX_CLASSES][3];
__constant__ int c_nc;

// kernel: maps 2D grid and 2D block into 1D global index
__global__ void classify_kernel(const uchar4_custom *input, uchar4_custom *output, int size) {
    // compute a linear thread index from 2D grid/block
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;
    unsigned int gdimx = gridDim.x;
    unsigned int bdimx = blockDim.x;
    unsigned int bdimy = blockDim.y;
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;

    unsigned long long blockLinear = (unsigned long long)by * gdimx + bx;
    unsigned long long threadsPerBlock = (unsigned long long)bdimx * bdimy;
    unsigned long long threadInBlock = (unsigned long long)ty * bdimx + tx;
    unsigned long long idx = blockLinear * threadsPerBlock + threadInBlock;

    if (idx >= (unsigned long long)size) return;

    float r = input[idx].x;
    float g = input[idx].y;
    float b = input[idx].z;

    float len_p = sqrtf(r * r + g * g + b * b);
    if (len_p < 1e-6f) len_p = 1.0f;

    float max_score = -1e9f;
    int best_class = 0;
    int nc = 0;
    // read c_nc from const memory (copied by host)
    asm volatile("" ::: "memory"); // small memory barrier hint (no-op) to be explicit

    // copy to local for speed (compiler will optimize)
    nc = c_nc;

    for (int j = 0; j < nc; ++j) {
        float dot = r * c_avg[j][0] + g * c_avg[j][1] + b * c_avg[j][2];
        float score = dot / len_p;
        if (score > max_score) {
            max_score = score;
            best_class = j;
        }
    }

    output[idx] = input[idx];
    output[idx].w = (uchar)best_class;
}

// CPU reference classification
void classifyCPU(const uchar4_custom *in, uchar4_custom *out, int size, float avg[MAX_CLASSES][3], int nc) {
    for (int i = 0; i < size; ++i) {
        float r = in[i].x;
        float g = in[i].y;
        float b = in[i].z;
        float len_p = sqrtf(r * r + g * g + b * b);
        if (len_p < 1e-6f) len_p = 1.0f;
        float max_score = -1e9f;
        int best = 0;
        for (int j = 0; j < nc; ++j) {
            float dot = r * avg[j][0] + g * avg[j][1] + b * avg[j][2];
            float sc = dot / len_p;
            if (sc > max_score) {
                max_score = sc;
                best = j;
            }
        }
        out[i] = in[i];
        out[i].w = (uchar)best;
    }
}

static void normalize_vector(float v[3]) {
    float len = sqrtf(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    if (len < 1e-6f) len = 1.0f;
    v[0] /= len; v[1] /= len; v[2] /= len;
}

int main() {
    srand((unsigned int)time(NULL));

    // test resolutions (as in example)
    int widths[]  = {300, 640, 2048};
    int heights[] = {300, 640, 2048};
    int numTests = 3;

    // grid and block configurations (to reproduce previous report style)
    int gridSizes[]  = {1, 8, 64, 512};
    int blockSizes[] = {32, 64, 128, 256, 512};

    for (int t = 0; t < numTests; ++t) {
        int w = widths[t], h = heights[t];
        printf("\n=== TEST %d (%dx%d) ===\n", t + 1, w, h);

        size_t size = (size_t)w * (size_t)h;
        printf("%zu\n", size);

        // generate random image
        uchar4_custom *host_img = (uchar4_custom*)malloc(size * sizeof(uchar4_custom));
        uchar4_custom *host_out_cpu = (uchar4_custom*)malloc(size * sizeof(uchar4_custom));
        uchar4_custom *host_out_gpu = (uchar4_custom*)malloc(size * sizeof(uchar4_custom));
        if (!host_img || !host_out_cpu || !host_out_gpu) {
            fprintf(stderr, "OOM\n");
            return 1;
        }
        for (size_t i = 0; i < size; ++i) {
            host_img[i].x = (uchar)(rand() % 256);
            host_img[i].y = (uchar)(rand() % 256);
            host_img[i].z = (uchar)(rand() % 256);
            host_img[i].w = 255;
        }

        // generate random class average vectors (normalized)
        int nc = (rand() % (MAX_CLASSES-1)) + 1; // 1..MAX_CLASSES-1 random for test
        float avg[MAX_CLASSES][3];
        for (int j = 0; j < nc; ++j) {
            avg[j][0] = (float)(rand() % 256);
            avg[j][1] = (float)(rand() % 256);
            avg[j][2] = (float)(rand() % 256);
            normalize_vector(avg[j]);
        }

        // copy to device constant memory
        CSC(cudaMemcpyToSymbol(c_avg, avg, sizeof(float) * MAX_CLASSES * 3));
        CSC(cudaMemcpyToSymbol(c_nc, &nc, sizeof(int)));

        // allocate device buffers
        uchar4_custom *d_in = NULL, *d_out = NULL;
        CSC(cudaMalloc(&d_in, size * sizeof(uchar4_custom)));
        CSC(cudaMalloc(&d_out, size * sizeof(uchar4_custom)));
        CSC(cudaMemcpy(d_in, host_img, size * sizeof(uchar4_custom), cudaMemcpyHostToDevice));

        printf("GPU timings (in ms):\n");

        // iterate configurations
        for (int gi = 0; gi < (int)(sizeof(gridSizes)/sizeof(gridSizes[0])); ++gi) {
            for (int bi = 0; bi < (int)(sizeof(blockSizes)/sizeof(blockSizes[0])); ++bi) {
                int g = gridSizes[gi];
                int b = blockSizes[bi];

                // map to 2D launch used in the previous report:
                // grid = (g, g)
                // block = (b/32, 32)  (so total threads per block = b)
                int block_x = b / 32;
                if (block_x < 1) block_x = 1;
                int block_y = 32;
                dim3 grid((unsigned int)g, (unsigned int)g);
                dim3 block((unsigned int)block_x, (unsigned int)block_y);

                // compute total threads launched; skip configs that produce zero useful threads
                unsigned long long threadsPerBlock = (unsigned long long)block.x * block.y;
                unsigned long long totalThreads = (unsigned long long)grid.x * grid.y * threadsPerBlock;
                if (threadsPerBlock == 0 || totalThreads == 0) {
                    printf("|grids: %d|blocks: %d|time: 0.000000 ms|\n", g, b);
                    continue;
                }

                // CUDA timing
                cudaEvent_t start, stop;
                CSC(cudaEventCreate(&start));
                CSC(cudaEventCreate(&stop));
                CSC(cudaEventRecord(start, 0));

                classify_kernel<<<grid, block>>>(d_in, d_out, (int)size);
                CSC(cudaGetLastError());
                CSC(cudaDeviceSynchronize());

                CSC(cudaEventRecord(stop, 0));
                CSC(cudaEventSynchronize(stop));
                float gpuTime = 0.0f;
                CSC(cudaEventElapsedTime(&gpuTime, start, stop));
                CSC(cudaEventDestroy(start));
                CSC(cudaEventDestroy(stop));

                // copy result back (to be thorough)
                CSC(cudaMemcpy(host_out_gpu, d_out, size * sizeof(uchar4_custom), cudaMemcpyDeviceToHost));
                printf("|grids: %d|blocks: %d|time: %.6f ms|\n", g, b, gpuTime);
            }
        }

        // CPU timing
        auto cpuStart = std::chrono::high_resolution_clock::now();
        classifyCPU(host_img, host_out_cpu, (int)size, avg, nc);
        auto cpuEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpuTime = cpuEnd - cpuStart;
        printf("CPU: %.6f ms\n", cpuTime.count());

        // clean
        CSC(cudaFree(d_in));
        CSC(cudaFree(d_out));
        free(host_img);
        free(host_out_cpu);
        free(host_out_gpu);
    }

    return 0;
}