#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_CLASSES 32

#define CSC(call)  									                \
do {											                    \
	cudaError_t res = call;							                \
	if (res != cudaSuccess) {							            \
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);								                    \
	}										                        \
} while(0)

__constant__ float c_avg[MAX_CLASSES][3]; // усредненные векторы классов
__constant__ int c_nc; // количество классов

typedef unsigned char uchar;

typedef struct {
    uchar x, y, z, w;
} uchar4_custom;

int loadImage(const char *fname, int *pw, int *ph, uchar4_custom **pdata) {
    FILE *f = fopen(fname, "rb");
    if (!f) {
        fprintf(stderr, "Не удалось открыть файл %s\n", fname);
        return 0;
    }

    unsigned int w = 0, h = 0;
    if (fread(&w, sizeof(unsigned int), 1, f) != 1 || fread(&h, sizeof(unsigned int), 1, f) != 1) {
        fprintf(stderr, "Ошибка чтения заголовка\n");
        fclose(f);
        return 0;
    }

    size_t count = (size_t)w * (size_t)h;
    uchar4_custom *buf = (uchar4_custom *)malloc(count * sizeof(uchar4_custom));
    if (!buf) {
        fprintf(stderr, "Ошибка выделения памяти\n");
        fclose(f);
        return 0;
    }

    size_t got = fread(buf, sizeof(uchar4_custom), count, f);
    fclose(f);
    if (got != count) {
        fprintf(stderr, "Ошибка чтения данных изображения\n");
        free(buf);
        return 0;
    }

    *pw = (int)w;
    *ph = (int)h;
    *pdata = buf;
    return 1;
}

int saveImage(const char *fname, int w, int h, const uchar4_custom *data) {
    FILE *f = fopen(fname, "wb");
    if (!f) {
        fprintf(stderr, "Не удалось создать файл %s\n", fname);
        return 0;
    }

    unsigned int ww = (unsigned int)w;
    unsigned int hh = (unsigned int)h;
    fwrite(&ww, sizeof(unsigned int), 1, f);
    fwrite(&hh, sizeof(unsigned int), 1, f);
    fwrite(data, sizeof(uchar4_custom), (size_t)w * (size_t)h, f);
    fclose(f);
    return 1;
}

__global__ void kernel(uchar4_custom *input, uchar4_custom *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) {
        return;
    }

    float r = input[idx].x;
    float g = input[idx].y;
    float b = input[idx].z;

    float len_p = sqrtf(r * r + g * g + b * b);
    if (len_p < 1e-6f) {
        len_p = 1.0f;
    }

    float max_score = -1e9f;
    int best_class = 0;

    for (int j = 0; j < c_nc; ++j) {
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

int main() {
    char inPath[256], outPath[256];
    
    if (!fgets(inPath, sizeof(inPath), stdin)) {
        fprintf(stderr, "Ошибка чтения пути входного файла\n");
        return 1;
    }
    if (!fgets(outPath, sizeof(outPath), stdin)) {
        fprintf(stderr, "Ошибка чтения пути выходного файла\n");
        return 1;
    }

    inPath[strcspn(inPath, "\r\n")] = '\0';
    outPath[strcspn(outPath, "\r\n")] = '\0';

    if (strlen(inPath) == 0 || strlen(outPath) == 0) {
        fprintf(stderr, "Пути не должны быть пустыми\n");
        return 1;
    }

    int w, h;
    uchar4_custom *data = NULL;
    if (!loadImage(inPath, &w, &h, &data)) {
        fprintf(stderr, "Ошибка загрузки входного файла: %s\n", inPath);
        return 1;
    }

    int nc;
    scanf("%d", &nc);
    if (nc > MAX_CLASSES) {
        fprintf(stderr, "Слишком много классов (максимум %d)\n", MAX_CLASSES);
        return 1;
    }
    
    float avg[MAX_CLASSES][3] = {0.0f};
    int np;
    for (int j = 0; j < nc; ++j) {
        scanf("%d", &np);
        if (np <= 0) {
            fprintf(stderr, "Ошибка: класс %d не содержит пикселей\n", j);
            return 1;
        }

        for (int k = 0; k < np; ++k) {
            int x, y;
            scanf("%d %d", &x, &y);
            if (x < 0 || x >= w || y < 0 || y >= h) {
                continue;
            }
            uchar4_custom p = data[y * w + x];
            avg[j][0] += p.x;
            avg[j][1] += p.y;
            avg[j][2] += p.z;
        }

        avg[j][0] /= np;
        avg[j][1] /= np;
        avg[j][2] /= np;

        float len = sqrtf(avg[j][0]*avg[j][0] + avg[j][1]*avg[j][1] + avg[j][2]*avg[j][2]);
        if (len < 1e-6f) len = 1.0f;
        avg[j][0] /= len;
        avg[j][1] /= len;
        avg[j][2] /= len;
    }

    CSC(cudaMemcpyToSymbol(c_avg, avg, nc * 3 * sizeof(float)));
    CSC(cudaMemcpyToSymbol(c_nc, &nc, sizeof(int)));

    size_t size = (size_t)w * (size_t)h;
    uchar4_custom *dev_in = NULL, *dev_out = NULL;
    CSC(cudaMalloc(&dev_in, size * sizeof(uchar4_custom)));
    CSC(cudaMalloc(&dev_out, size * sizeof(uchar4_custom)));
    CSC(cudaMemcpy(dev_in, data, size * sizeof(uchar4_custom), cudaMemcpyHostToDevice));

    kernel<<<400000, 1024>>>(dev_in, dev_out, (int)size);
    CSC(cudaGetLastError());
    CSC(cudaMemcpy(data, dev_out, size * sizeof(uchar4_custom), cudaMemcpyDeviceToHost));

    if (!saveImage(outPath, w, h, data)) {
        fprintf(stderr, "Ошибка записи результата\n");
        free(data);
        return 1;
    }

    CSC(cudaFree(dev_in));
    CSC(cudaFree(dev_out));
    free(data);
    return 0;
}