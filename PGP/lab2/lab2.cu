#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CSC(call)  									                \
do {											                    \
	cudaError_t res = call;							                \
	if (res != cudaSuccess) {							            \
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);								                    \
	}										                        \
} while(0)

int loadImage(const char *fname, int *pw, int *ph, uchar4 **pdata) {
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
    uchar4 *buf = (uchar4 *)malloc(count * sizeof(uchar4));
    if (!buf) {
        fprintf(stderr, "Ошибка выделения памяти\n");
        fclose(f);
        return 0;
    }

    size_t got = fread(buf, sizeof(uchar4), count, f);
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

int saveImage(const char *fname, int w, int h, const uchar4 *data) {
    FILE *f = fopen(fname, "wb");
    if (!f) {
        fprintf(stderr, "Не удалось создать файл %s\n", fname);
        return 0;
    }

    unsigned int ww = (unsigned int)w;
    unsigned int hh = (unsigned int)h;
    fwrite(&ww, sizeof(unsigned int), 1, f);
    fwrite(&hh, sizeof(unsigned int), 1, f);
    fwrite(data, sizeof(uchar4), (size_t)w * (size_t)h, f);
    fclose(f);
    return 1;
}

__global__ void kernel(cudaTextureObject_t tex, uchar4 *out, int w, int h) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

   	int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;
    int x, y;

    const float cR = 0.299f;
    const float cG = 0.587f;
    const float cB = 0.114f;

    for(y = idy; y < h; y += offsety) {
        for(x = idx; x < w; x += offsetx) {
            float fx = (x + 0.5f) / (float)w;
            float fy = (y + 0.5f) / (float)h;

            float Y00, Y10, Y20, Y01, Y11, Y21, Y02, Y12, Y22;
            uchar4 p;

            // row y-1
            p = tex2D<uchar4>(tex, (fx - 1.0f / w), (fy - 1.0f / h));
            Y00 = cR*p.x + cG*p.y + cB*p.z;
            p = tex2D<uchar4>(tex, fx, (fy - 1.0f / h));
            Y10 = cR*p.x + cG*p.y + cB*p.z;
            p = tex2D<uchar4>(tex, (fx + 1.0f / w), (fy - 1.0f / h));
            Y20 = cR*p.x + cG*p.y + cB*p.z;

            // row y
            p = tex2D<uchar4>(tex, (fx - 1.0f / w), fy);
            Y01 = cR*p.x + cG*p.y + cB*p.z;
            p = tex2D<uchar4>(tex, fx, fy);
            Y11 = cR*p.x + cG*p.y + cB*p.z;
            unsigned char alpha = p.w;
            p = tex2D<uchar4>(tex, (fx + 1.0f / w), fy);
            Y21 = cR*p.x + cG*p.y + cB*p.z;

            // row y+1
            p = tex2D<uchar4>(tex, (fx - 1.0f / w), (fy + 1.0f / h));
            Y02 = cR*p.x + cG*p.y + cB*p.z;
            p = tex2D<uchar4>(tex, fx, (fy + 1.0f / h));
            Y12 = cR*p.x + cG*p.y + cB*p.z;
            p = tex2D<uchar4>(tex, (fx + 1.0f / w), (fy + 1.0f / h));
            Y22 = cR*p.x + cG*p.y + cB*p.z;

            float Gx = (-1.0f * Y00) + (0.0f * Y10) + (1.0f * Y20)
                      + (-2.0f * Y01) + (0.0f * Y11) + (2.0f * Y21)
                      + (-1.0f * Y02) + (0.0f * Y12) + (1.0f * Y22);
            
            float Gy = (-1.0f * Y00) + (-2.0f * Y10) + (-1.0f * Y20)
                      + ( 0.0f * Y01) + ( 0.0f * Y11) + ( 0.0f * Y21)
                      + ( 1.0f * Y02) + ( 2.0f * Y12) + ( 1.0f * Y22);

            float mag = sqrtf(Gx * Gx + Gy * Gy);
            int val = (int)(mag + 0.5f);
            if (val > 255) {
                val = 255;
            }
            if (val < 0) {
                val = 0;
            }

            out[y * w + x] = make_uchar4(val, val, val, alpha);
        }
    }
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
    uchar4 *data = NULL;
    if (!loadImage(inPath, &w, &h, &data)) {
        fprintf(stderr, "Ошибка загрузки входного файла: %s\n", inPath);
        return 1;
    }

    cudaArray *arr;
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
    CSC(cudaMallocArray(&arr, &ch, w, h));
    CSC(cudaMemcpy2DToArray(arr, 0, 0, data, w * sizeof(uchar4), w * sizeof(uchar4), h, cudaMemcpyHostToDevice));

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = arr;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = true;

    cudaTextureObject_t tex = 0;
    CSC(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));

    uchar4 *dev_out;
    CSC(cudaMalloc(&dev_out, sizeof(uchar4) * (size_t)w * (size_t)h));

    kernel<<< dim3(16, 16), dim3(32, 32) >>>(tex, dev_out, w, h);
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * (size_t)w * (size_t)h, cudaMemcpyDeviceToHost));

    CSC(cudaDestroyTextureObject(tex));
    CSC(cudaFreeArray(arr));
    CSC(cudaFree(dev_out));

    if (!saveImage(outPath, w, h, data)) {
        fprintf(stderr, "Ошибка записи результата\n");
        free(data);
        return 1;
    }

    free(data);
    return 0;
}