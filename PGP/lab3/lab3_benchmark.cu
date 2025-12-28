#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <cuda_runtime.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CSC(call) \
do { \
    cudaError_t res = call; \
    if (res != cudaSuccess) { \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n", __FILE__, __LINE__, cudaGetErrorString(res)); \
        exit(0); \
    } \
} while(0)

typedef unsigned char uchar;

struct vec3 { double x, y, z; };
struct trig { vec3 a, b, c; uchar4 color; };

__host__ __device__ double dot(vec3 a, vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ vec3 prod(vec3 a, vec3 b) {
    vec3 result;
    result.x = a.y * b.z - a.z * b.y;
    result.y = a.z * b.x - a.x * b.z;
    result.z = a.x * b.y - a.y * b.x;
    return result;
}

__host__ __device__ vec3 norm(vec3 v) {
    double l = sqrt(dot(v, v));
    vec3 result;
    if (l < 1e-10) {
        result.x = result.y = result.z = 0;
    } else {
        result.x = v.x / l;
        result.y = v.y / l;
        result.z = v.z / l;
    }
    return result;
}

__host__ __device__ vec3 diff(vec3 a, vec3 b) {
    vec3 result;
    result.x = a.x - b.x;
    result.y = a.y - b.y;
    result.z = a.z - b.z;
    return result;
}

__host__ __device__ vec3 add(vec3 a, vec3 b) {
    vec3 result;
    result.x = a.x + b.x;
    result.y = a.y + b.y;
    result.z = a.z + b.z;
    return result;
}

__host__ __device__ vec3 mult(vec3 a, vec3 b, vec3 c, vec3 v) {
    vec3 result;
    result.x = a.x * v.x + b.x * v.y + c.x * v.z;
    result.y = a.y * v.x + b.y * v.y + c.y * v.z;
    result.z = a.z * v.x + b.z * v.y + c.z * v.z;
    return result;
}

__host__ __device__ vec3 scale(vec3 v, double s) {
    vec3 result;
    result.x = v.x * s;
    result.y = v.y * s;
    result.z = v.z * s;
    return result;
}

__host__ __device__ bool intersect_triangle(vec3 pos, vec3 dir, trig t, double* ts) {
    vec3 e1 = diff(t.b, t.a);
    vec3 e2 = diff(t.c, t.a);
    vec3 p = prod(dir, e2);
    double div = dot(p, e1);
    
    if (fabs(div) < 1e-10) return false;
    
    vec3 tv = diff(pos, t.a);
    double u = dot(p, tv) / div;
    if (u < 0.0 || u > 1.0) return false;
    
    vec3 q = prod(tv, e1);
    double v = dot(q, dir) / div;
    if (v < 0.0 || v + u > 1.0) return false;
    
    double t_val = dot(q, e2) / div;
    if (t_val < 1e-6) return false;
    
    *ts = t_val;
    return true;
}

__global__ void render_kernel(vec3 pc, vec3 pv, int w, int h, double angle, 
                              trig* trigs, int num_trigs, vec3 light_pos, uchar4* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx >= w || idy >= h) return;
    
    double dw = 2.0 / (w - 1);
    double dh = 2.0 / (h - 1);
    double z = 1.0 / tan(angle * M_PI / 360.0);
    
    vec3 up_vec;
    up_vec.x = 0.0; up_vec.y = 1.0; up_vec.z = 0.0;
    
    vec3 bz = norm(diff(pv, pc));
    vec3 bx = norm(prod(bz, up_vec));
    vec3 by = prod(bx, bz);
    
    vec3 v;
    v.x = -1.0 + dw * idx;
    v.y = (-1.0 + dh * idy) * h / w;
    v.z = z;
    vec3 dir = norm(mult(bx, by, bz, v));
    
    int k_min = -1;
    double ts_min = 1e20;
    
    for (int k = 0; k < num_trigs; k++) {
        double ts;
        if (intersect_triangle(pc, dir, trigs[k], &ts)) {
            if (ts < ts_min) {
                k_min = k;
                ts_min = ts;
            }
        }
    }
    
    uchar4 color;
    if (k_min == -1) {
        color = make_uchar4(135, 206, 235, 0);
    } else {
        vec3 hit_point = add(pc, scale(dir, ts_min));
        vec3 e1 = diff(trigs[k_min].b, trigs[k_min].a);
        vec3 e2 = diff(trigs[k_min].c, trigs[k_min].a);
        vec3 normal = norm(prod(e1, e2));
        
        vec3 to_light = norm(diff(light_pos, hit_point));
        double diffuse = fmax(0.3, dot(normal, to_light));
        
        color = trigs[k_min].color;
        color.x = (uchar)(color.x * diffuse);
        color.y = (uchar)(color.y * diffuse);
        color.z = (uchar)(color.z * diffuse);
    }
    
    data[(h - 1 - idy) * w + idx] = color;
}

void render_cpu(vec3 pc, vec3 pv, int w, int h, double angle, 
                trig* trigs, int num_trigs, vec3 light_pos, uchar4* data) {
    double dw = 2.0 / (w - 1);
    double dh = 2.0 / (h - 1);
    double z = 1.0 / tan(angle * M_PI / 360.0);
    
    vec3 up_vec;
    up_vec.x = 0.0; up_vec.y = 1.0; up_vec.z = 0.0;
    
    vec3 bz = norm(diff(pv, pc));
    vec3 bx = norm(prod(bz, up_vec));
    vec3 by = prod(bx, bz);
    
    for (int i = 0; i < w; i++) {
        for (int j = 0; j < h; j++) {
            vec3 v;
            v.x = -1.0 + dw * i;
            v.y = (-1.0 + dh * j) * h / w;
            v.z = z;
            vec3 dir = norm(mult(bx, by, bz, v));
            
            int k_min = -1;
            double ts_min = 1e20;
            
            for (int k = 0; k < num_trigs; k++) {
                double ts;
                if (intersect_triangle(pc, dir, trigs[k], &ts)) {
                    if (ts < ts_min) {
                        k_min = k;
                        ts_min = ts;
                    }
                }
            }
            
            uchar4 color;
            if (k_min == -1) {
                color = make_uchar4(135, 206, 235, 0);
            } else {
                vec3 hit_point = add(pc, scale(dir, ts_min));
                vec3 e1 = diff(trigs[k_min].b, trigs[k_min].a);
                vec3 e2 = diff(trigs[k_min].c, trigs[k_min].a);
                vec3 normal = norm(prod(e1, e2));
                
                vec3 to_light = norm(diff(light_pos, hit_point));
                double diffuse = fmax(0.3, dot(normal, to_light));
                
                color = trigs[k_min].color;
                color.x = (uchar)(color.x * diffuse);
                color.y = (uchar)(color.y * diffuse);
                color.z = (uchar)(color.z * diffuse);
            }
            
            data[(h - 1 - j) * w + i] = color;
        }
    }
}

void add_tetrahedron(trig* trigs, int* idx, vec3 center, double radius, uchar4 color) {
    double a = radius * sqrt(8.0/3.0);
    double h = a * sqrt(2.0/3.0);
    
    vec3 v0 = add(center, (vec3){0, h/2, 0});
    vec3 v1 = add(center, (vec3){-a/2, -h/2, a/(2*sqrt(3))});
    vec3 v2 = add(center, (vec3){a/2, -h/2, a/(2*sqrt(3))});
    vec3 v3 = add(center, (vec3){0, -h/2, -a/sqrt(3)});
    
    trigs[(*idx)++] = (trig){v0, v1, v2, color};
    trigs[(*idx)++] = (trig){v0, v2, v3, color};
    trigs[(*idx)++] = (trig){v0, v3, v1, color};
    trigs[(*idx)++] = (trig){v1, v3, v2, color};
}

void add_hexahedron(trig* trigs, int* idx, vec3 center, double radius, uchar4 color) {
    double a = radius * 2.0 / sqrt(3.0);
    double h = a / 2.0;
    
    vec3 vertices[8] = {
        add(center, (vec3){-h, -h, -h}), add(center, (vec3){h, -h, -h}),
        add(center, (vec3){h, h, -h}), add(center, (vec3){-h, h, -h}),
        add(center, (vec3){-h, -h, h}), add(center, (vec3){h, -h, h}),
        add(center, (vec3){h, h, h}), add(center, (vec3){-h, h, h})
    };
    
    int faces[12][3] = {
        {0,1,2}, {0,2,3}, {4,6,5}, {4,7,6},
        {0,4,5}, {0,5,1}, {2,6,7}, {2,7,3},
        {0,3,7}, {0,7,4}, {1,5,6}, {1,6,2}
    };
    
    for (int i = 0; i < 12; i++) {
        trigs[(*idx)++] = (trig){vertices[faces[i][0]], vertices[faces[i][1]], 
                                 vertices[faces[i][2]], color};
    }
}

void add_dodecahedron(trig* trigs, int* idx, vec3 center, double radius, uchar4 color) {
    double phi = (1.0 + sqrt(5.0)) / 2.0;
    double a = radius / sqrt(3.0);
    
    vec3 v[20];
    v[0] = add(center, scale((vec3){1, 1, 1}, a));
    v[1] = add(center, scale((vec3){1, 1, -1}, a));
    v[2] = add(center, scale((vec3){1, -1, 1}, a));
    v[3] = add(center, scale((vec3){1, -1, -1}, a));
    v[4] = add(center, scale((vec3){-1, 1, 1}, a));
    v[5] = add(center, scale((vec3){-1, 1, -1}, a));
    v[6] = add(center, scale((vec3){-1, -1, 1}, a));
    v[7] = add(center, scale((vec3){-1, -1, -1}, a));
    
    v[8] = add(center, scale((vec3){0, phi, 1.0/phi}, a));
    v[9] = add(center, scale((vec3){0, phi, -1.0/phi}, a));
    v[10] = add(center, scale((vec3){0, -phi, 1.0/phi}, a));
    v[11] = add(center, scale((vec3){0, -phi, -1.0/phi}, a));
    
    v[12] = add(center, scale((vec3){1.0/phi, 0, phi}, a));
    v[13] = add(center, scale((vec3){-1.0/phi, 0, phi}, a));
    v[14] = add(center, scale((vec3){1.0/phi, 0, -phi}, a));
    v[15] = add(center, scale((vec3){-1.0/phi, 0, -phi}, a));
    
    v[16] = add(center, scale((vec3){phi, 1.0/phi, 0}, a));
    v[17] = add(center, scale((vec3){phi, -1.0/phi, 0}, a));
    v[18] = add(center, scale((vec3){-phi, 1.0/phi, 0}, a));
    v[19] = add(center, scale((vec3){-phi, -1.0/phi, 0}, a));
    
    int faces[12][5] = {
        {0,8,9,1,16}, {0,12,13,4,8}, {0,16,17,2,12},
        {8,4,18,5,9}, {12,2,10,6,13}, {16,1,14,3,17},
        {9,5,15,14,1}, {6,10,11,7,19}, {4,13,6,19,18},
        {2,17,3,11,10}, {5,18,19,7,15}, {3,14,15,7,11}
    };
    
    for (int i = 0; i < 12; i++) {
        trigs[(*idx)++] = (trig){v[faces[i][0]], v[faces[i][1]], v[faces[i][2]], color};
        trigs[(*idx)++] = (trig){v[faces[i][0]], v[faces[i][2]], v[faces[i][3]], color};
        trigs[(*idx)++] = (trig){v[faces[i][0]], v[faces[i][3]], v[faces[i][4]], color};
    }
}

void build_scene(trig* scene, int* trig_count, int complexity) {
    *trig_count = 0;
    
    vec3 floor_v[4] = {
        {-8.0, -2.0, -8.0}, {-8.0, -2.0, 8.0},
        {8.0, -2.0, 8.0}, {8.0, -2.0, -8.0}
    };
    uchar4 floor_col = make_uchar4(178, 178, 178, 0);
    scene[(*trig_count)++] = (trig){floor_v[0], floor_v[1], floor_v[2], floor_col};
    scene[(*trig_count)++] = (trig){floor_v[0], floor_v[2], floor_v[3], floor_col};
    
    if (complexity >= 1) {
        add_tetrahedron(scene, trig_count, (vec3){-2.5, 0, 0}, 1.2, make_uchar4(255, 51, 51, 0));
    }
    if (complexity >= 2) {
        add_hexahedron(scene, trig_count, (vec3){0, 1.2, 0}, 1.0, make_uchar4(51, 255, 51, 0));
    }
    if (complexity >= 3) {
        add_dodecahedron(scene, trig_count, (vec3){2.5, 0, 0}, 1.2, make_uchar4(51, 51, 255, 0));
    }
}

int main() {
    printf("\n======================================================================\n");
    printf("ИССЛЕДОВАНИЕ ПРОИЗВОДИТЕЛЬНОСТИ RAY TRACING\n");
    printf("======================================================================\n\n");
    
    // TEST 1: Влияние конфигурации grid/block
    printf("=== ЧАСТЬ 1: Влияние конфигурации kernel на производительность ===\n\n");
    
    int test_widths[]  = {300, 640, 2048};
    int test_heights[] = {300, 640, 2048};
    int grid_sizes[] = {1, 8, 64, 256};
    int block_sizes[] = {32, 64, 128, 256, 512};
    
    for (int t = 0; t < 3; t++) {
        int w = test_widths[t], h = test_heights[t];
        size_t pixels = (size_t)w * h;
        
        printf("TEST %d (%dx%d)\n", t + 1, w, h);
        printf("%zu\n", pixels);
        
        const int MAX_TRIGS = 1000;
        trig* scene = (trig*)malloc(sizeof(trig) * MAX_TRIGS);
        int trig_count = 0;
        build_scene(scene, &trig_count, 3);
        
        vec3 pc = {8.0, 4.5, 0.0};
        vec3 pv = {0.0, 0.8, 0.0};
        vec3 light = {8.0, 15.0, 8.0};
        double fov = 120.0;
        
        uchar4* data = (uchar4*)malloc(sizeof(uchar4) * pixels);
        trig* d_scene;
        uchar4* d_data;
        
        CSC(cudaMalloc(&d_scene, sizeof(trig) * trig_count));
        CSC(cudaMalloc(&d_data, sizeof(uchar4) * pixels));
        CSC(cudaMemcpy(d_scene, scene, sizeof(trig) * trig_count, cudaMemcpyHostToDevice));
        
        printf("GPU timings (in ms):\n");
        
        for (int gi = 0; gi < 4; gi++) {
            for (int bi = 0; bi < 5; bi++) {
                int g = grid_sizes[gi];
                int b = block_sizes[bi];
                
                int block_x = b / 32;
                if (block_x < 1) block_x = 1;
                int block_y = 32;
                
                dim3 grid((unsigned int)g, (unsigned int)g);
                dim3 block((unsigned int)block_x, (unsigned int)block_y);
                
                cudaEvent_t start, stop;
                CSC(cudaEventCreate(&start));
                CSC(cudaEventCreate(&stop));
                CSC(cudaEventRecord(start, 0));
                
                render_kernel<<<grid, block>>>(pc, pv, w, h, fov, d_scene, trig_count, light, d_data);
                CSC(cudaGetLastError());
                CSC(cudaDeviceSynchronize());
                
                CSC(cudaEventRecord(stop, 0));
                CSC(cudaEventSynchronize(stop));
                
                float gpuTime = 0.0f;
                CSC(cudaEventElapsedTime(&gpuTime, start, stop));
                
                printf("|grids: %d|blocks: %d|time: %.6f ms|\n", g, b, gpuTime);
                
                CSC(cudaEventDestroy(start));
                CSC(cudaEventDestroy(stop));
            }
        }
        
        auto cpuStart = std::chrono::high_resolution_clock::now();
        render_cpu(pc, pv, w, h, fov, scene, trig_count, light, data);
        auto cpuEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpuTime = cpuEnd - cpuStart;
        printf("CPU: %.6f ms\n", cpuTime.count());
        
        CSC(cudaFree(d_scene));
        CSC(cudaFree(d_data));
        free(data);
        free(scene);
        printf("\n");
    }
    
    // TEST 2: Влияние сложности сцены
    printf("\n=== ЧАСТЬ 2: Влияние сложности сцены (кол-во треугольников) ===\n\n");
    
    int w = 640, h = 640;
    size_t pixels = (size_t)w * h;
    vec3 pc = {8.0, 4.5, 0.0};
    vec3 pv = {0.0, 0.8, 0.0};
    vec3 light = {8.0, 15.0, 8.0};
    double fov = 120.0;
    
    const char* complexity_names[] = {"Только пол (2 треугольника)", 
                                       "Пол + Тетраэдр (6 треугольников)",
                                       "Пол + Тетраэдр + Гексаэдр (18 треугольников)",
                                       "Полная сцена (54 треугольника)"};
    
    for (int complexity = 0; complexity <= 3; complexity++) {
        const int MAX_TRIGS = 1000;
        trig* scene = (trig*)malloc(sizeof(trig) * MAX_TRIGS);
        int trig_count = 0;
        build_scene(scene, &trig_count, complexity);
        
        printf("Сцена: %s\n", complexity_names[complexity]);
        
        uchar4* data = (uchar4*)malloc(sizeof(uchar4) * pixels);
        trig* d_scene;
        uchar4* d_data;
        
        CSC(cudaMalloc(&d_scene, sizeof(trig) * trig_count));
        CSC(cudaMalloc(&d_data, sizeof(uchar4) * pixels));
        CSC(cudaMemcpy(d_scene, scene, sizeof(trig) * trig_count, cudaMemcpyHostToDevice));
        
        dim3 block(16, 16);
        dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
        
        cudaEvent_t start, stop;
        CSC(cudaEventCreate(&start));
        CSC(cudaEventCreate(&stop));
        CSC(cudaEventRecord(start, 0));
        
        render_kernel<<<grid, block>>>(pc, pv, w, h, fov, d_scene, trig_count, light, d_data);
        CSC(cudaGetLastError());
        CSC(cudaDeviceSynchronize());
        
        CSC(cudaEventRecord(stop, 0));
        CSC(cudaEventSynchronize(stop));
        
        float gpuTime = 0.0f;
        CSC(cudaEventElapsedTime(&gpuTime, start, stop));
        
        auto cpuStart = std::chrono::high_resolution_clock::now();
        render_cpu(pc, pv, w, h, fov, scene, trig_count, light, data);
        auto cpuEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpuTime = cpuEnd - cpuStart;
        
        printf("GPU: %.6f ms | CPU: %.6f ms | Ускорение: %.2fx\n\n", 
               gpuTime, cpuTime.count(), cpuTime.count() / gpuTime);
        
        CSC(cudaEventDestroy(start));
        CSC(cudaEventDestroy(stop));
        CSC(cudaFree(d_scene));
        CSC(cudaFree(d_data));
        free(data);
        free(scene);
    }
    
    // TEST 3: Влияние ракурса камеры
    printf("\n=== ЧАСТЬ 3: Влияние ракурса камеры ===\n\n");
    
    const int MAX_TRIGS = 1000;
    trig* scene = (trig*)malloc(sizeof(trig) * MAX_TRIGS);
    int trig_count = 0;
    build_scene(scene, &trig_count, 3);
    
    vec3 cameras[] = {
        {8.0, 4.5, 0.0},    // Фронтальный вид
        {0.0, 10.0, 8.0},   // Вид сверху
        {3.0, 1.0, 3.0},    // Близкий ракурс
        {15.0, 5.0, 15.0}   // Дальний ракурс
    };
    
    const char* camera_names[] = {"Фронтальный вид", "Вид сверху", "Близкий ракурс", "Дальний ракурс"};
    
    for (int cam_idx = 0; cam_idx < 4; cam_idx++) {
        vec3 pc = cameras[cam_idx];
        vec3 pv = {0.0, 0.8, 0.0};
        
        printf("Ракурс: %s (камера: %.1f, %.1f, %.1f)\n", 
               camera_names[cam_idx], pc.x, pc.y, pc.z);
        
        uchar4* data = (uchar4*)malloc(sizeof(uchar4) * pixels);
        trig* d_scene;
        uchar4* d_data;
        
        CSC(cudaMalloc(&d_scene, sizeof(trig) * trig_count));
        CSC(cudaMalloc(&d_data, sizeof(uchar4) * pixels));
        CSC(cudaMemcpy(d_scene, scene, sizeof(trig) * trig_count, cudaMemcpyHostToDevice));
        
        dim3 block(16, 16);
        dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
        
        cudaEvent_t start, stop;
        CSC(cudaEventCreate(&start));
        CSC(cudaEventCreate(&stop));
        CSC(cudaEventRecord(start, 0));
        
        render_kernel<<<grid, block>>>(pc, pv, w, h, fov, d_scene, trig_count, light, d_data);
        CSC(cudaGetLastError());
        CSC(cudaDeviceSynchronize());
        
        CSC(cudaEventRecord(stop, 0));
        CSC(cudaEventSynchronize(stop));
        
        float gpuTime = 0.0f;
        CSC(cudaEventElapsedTime(&gpuTime, start, stop));
        
        printf("GPU: %.6f ms\n\n", gpuTime);
        
        CSC(cudaEventDestroy(start));
        CSC(cudaEventDestroy(stop));
        CSC(cudaFree(d_scene));
        CSC(cudaFree(d_data));
        free(data);
    }
    
    free(scene);
    
    printf("======================================================================\n");
    printf("БЕНЧМАРК ЗАВЕРШЁН\n");
    printf("======================================================================\n");
    
    return 0;
}