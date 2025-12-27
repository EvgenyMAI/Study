#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef unsigned char uchar;

struct vec3 {
    double x, y, z;
};

struct trig {
    vec3 a, b, c;
    uchar4 color;
};

__host__ __device__ double dot(vec3 a, vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ vec3 prod(vec3 a, vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

__host__ __device__ vec3 norm(vec3 v) {
    double l = sqrt(dot(v, v));
    if (l < 1e-10) return {0, 0, 0};
    return {v.x / l, v.y / l, v.z / l};
}

__host__ __device__ vec3 diff(vec3 a, vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__host__ __device__ vec3 add(vec3 a, vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__host__ __device__ vec3 mult(vec3 a, vec3 b, vec3 c, vec3 v) {
    return {
        a.x * v.x + b.x * v.y + c.x * v.z,
        a.y * v.x + b.y * v.y + c.y * v.z,
        a.z * v.x + b.z * v.y + c.z * v.z
    };
}

__host__ __device__ vec3 scale(vec3 v, double s) {
    return {v.x * s, v.y * s, v.z * s};
}

__host__ __device__ bool intersect_triangle(vec3 pos, vec3 dir, trig t, double& ts) {
    vec3 e1 = diff(t.b, t.a);
    vec3 e2 = diff(t.c, t.a);
    vec3 p = prod(dir, e2);
    double div = dot(p, e1);
    
    if (fabs(div) < 1e-10)
        return false;
    
    vec3 tv = diff(pos, t.a);
    double u = dot(p, tv) / div;
    if (u < 0.0 || u > 1.0)
        return false;
    
    vec3 q = prod(tv, e1);
    double v = dot(q, dir) / div;
    if (v < 0.0 || v + u > 1.0)
        return false;
    
    double t_val = dot(q, e2) / div;
    if (t_val < 1e-6)
        return false;
    
    ts = t_val;
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
    
    vec3 bz = norm(diff(pv, pc));
    vec3 bx = norm(prod(bz, {0.0, 1.0, 0.0}));
    vec3 by = prod(bx, bz);
    
    vec3 v = {-1.0 + dw * idx, (-1.0 + dh * idy) * h / w, z};
    vec3 dir = norm(mult(bx, by, bz, v));
    
    int k_min = -1;
    double ts_min = 1e20;
    
    for (int k = 0; k < num_trigs; k++) {
        double ts;
        if (intersect_triangle(pc, dir, trigs[k], ts)) {
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
    
    vec3 bz = norm(diff(pv, pc));
    vec3 bx = norm(prod(bz, {0.0, 1.0, 0.0}));
    vec3 by = prod(bx, bz);
    
    for (int i = 0; i < w; i++) {
        for (int j = 0; j < h; j++) {
            vec3 v = {-1.0 + dw * i, (-1.0 + dh * j) * h / w, z};
            vec3 dir = norm(mult(bx, by, bz, v));
            
            int k_min = -1;
            double ts_min = 1e20;
            
            for (int k = 0; k < num_trigs; k++) {
                double ts;
                if (intersect_triangle(pc, dir, trigs[k], ts)) {
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

void add_tetrahedron(trig* trigs, int& idx, vec3 center, double radius, uchar4 color) {
    double a = radius * sqrt(8.0/3.0);
    double h = a * sqrt(2.0/3.0);
    
    vec3 v0 = add(center, {0, h/2, 0});
    vec3 v1 = add(center, {-a/2, -h/2, a/(2*sqrt(3))});
    vec3 v2 = add(center, {a/2, -h/2, a/(2*sqrt(3))});
    vec3 v3 = add(center, {0, -h/2, -a/sqrt(3)});
    
    trigs[idx++] = {v0, v1, v2, color};
    trigs[idx++] = {v0, v2, v3, color};
    trigs[idx++] = {v0, v3, v1, color};
    trigs[idx++] = {v1, v3, v2, color};
}

void add_hexahedron(trig* trigs, int& idx, vec3 center, double radius, uchar4 color) {
    double a = radius * 2.0 / sqrt(3.0);
    double h = a / 2.0;
    
    vec3 vertices[8] = {
        add(center, {-h, -h, -h}), add(center, {h, -h, -h}),
        add(center, {h, h, -h}), add(center, {-h, h, -h}),
        add(center, {-h, -h, h}), add(center, {h, -h, h}),
        add(center, {h, h, h}), add(center, {-h, h, h})
    };
    
    int faces[12][3] = {
        {0,1,2}, {0,2,3}, {4,6,5}, {4,7,6},
        {0,4,5}, {0,5,1}, {2,6,7}, {2,7,3},
        {0,3,7}, {0,7,4}, {1,5,6}, {1,6,2}
    };
    
    for (int i = 0; i < 12; i++) {
        trigs[idx++] = {vertices[faces[i][0]], vertices[faces[i][1]], 
                        vertices[faces[i][2]], color};
    }
}

void add_dodecahedron(trig* trigs, int& idx, vec3 center, double radius, uchar4 color) {
    double phi = (1.0 + sqrt(5.0)) / 2.0;
    double a = radius / sqrt(3.0);
    
    vec3 v[20];
    v[0] = add(center, scale({1, 1, 1}, a));
    v[1] = add(center, scale({1, 1, -1}, a));
    v[2] = add(center, scale({1, -1, 1}, a));
    v[3] = add(center, scale({1, -1, -1}, a));
    v[4] = add(center, scale({-1, 1, 1}, a));
    v[5] = add(center, scale({-1, 1, -1}, a));
    v[6] = add(center, scale({-1, -1, 1}, a));
    v[7] = add(center, scale({-1, -1, -1}, a));
    
    v[8] = add(center, scale({0, phi, 1.0/phi}, a));
    v[9] = add(center, scale({0, phi, -1.0/phi}, a));
    v[10] = add(center, scale({0, -phi, 1.0/phi}, a));
    v[11] = add(center, scale({0, -phi, -1.0/phi}, a));
    
    v[12] = add(center, scale({1.0/phi, 0, phi}, a));
    v[13] = add(center, scale({-1.0/phi, 0, phi}, a));
    v[14] = add(center, scale({1.0/phi, 0, -phi}, a));
    v[15] = add(center, scale({-1.0/phi, 0, -phi}, a));
    
    v[16] = add(center, scale({phi, 1.0/phi, 0}, a));
    v[17] = add(center, scale({phi, -1.0/phi, 0}, a));
    v[18] = add(center, scale({-phi, 1.0/phi, 0}, a));
    v[19] = add(center, scale({-phi, -1.0/phi, 0}, a));
    
    int faces[12][5] = {
        {0,8,9,1,16}, {0,12,13,4,8}, {0,16,17,2,12},
        {8,4,18,5,9}, {12,2,10,6,13}, {16,1,14,3,17},
        {9,5,15,14,1}, {6,10,11,7,19}, {4,13,6,19,18},
        {2,17,3,11,10}, {5,18,19,7,15}, {3,14,15,7,11}
    };
    
    for (int i = 0; i < 12; i++) {
        trigs[idx++] = {v[faces[i][0]], v[faces[i][1]], v[faces[i][2]], color};
        trigs[idx++] = {v[faces[i][0]], v[faces[i][2]], v[faces[i][3]], color};
        trigs[idx++] = {v[faces[i][0]], v[faces[i][3]], v[faces[i][4]], color};
    }
}

int main(int argc, char** argv) {
    bool use_gpu = true;
    bool default_mode = false;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--cpu") == 0) use_gpu = false;
        else if (strcmp(argv[i], "--gpu") == 0) use_gpu = true;
        else if (strcmp(argv[i], "--default") == 0) default_mode = true;
    }
    
    if (default_mode) {
        printf("150\n");
        printf("out/img_%%d.data\n");
        printf("1280 720 120\n");
        printf("8.0 4.5 0.0 1.0 0.6 2.0 3.0 1.0 0.0 0.0\n");
        printf("0.0 0.8 0.0 0.0 0.3 0.0 2.0 0.0 0.0 0.0\n");
        printf("-2.5 0.0 0.0 1.0 0.2 0.2 1.2 0.9 0.1 0\n");
        printf("0.0 1.2 0.0 0.2 1.0 0.2 1.0 0.8 0.2 0\n");
        printf("2.5 0.0 0.0 0.2 0.2 1.0 1.2 0.7 0.3 0\n");
        printf("-8.0 -2.0 -8.0 -8.0 -2.0 8.0 8.0 -2.0 8.0 8.0 -2.0 -8.0\n");
        printf("floor.data 0.7 0.7 0.7 0.0\n");
        printf("1\n");
        printf("8.0 15.0 8.0 1.0 1.0 1.0\n");
        printf("1 1\n");
        return 0;
    }
    
    int frames;
    char output_path[256];
    int w, h;
    double fov;
    
    scanf("%d", &frames);
    scanf("%s", output_path);
    scanf("%d %d %lf", &w, &h, &fov);
    
    double cam_params[20];
    for (int i = 0; i < 20; i++) scanf("%lf", &cam_params[i]);
    
    double body_params[3][10];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 10; j++)
            scanf("%lf", &body_params[i][j]);
    
    char floor_texture[256];
    double floor_coords[12], floor_color[3], floor_reflect;
    for (int i = 0; i < 12; i++) scanf("%lf", &floor_coords[i]);
    scanf("%s", floor_texture);
    for (int i = 0; i < 3; i++) scanf("%lf", &floor_color[i]);
    scanf("%lf", &floor_reflect);
    
    int num_lights;
    scanf("%d", &num_lights);
    double light_params[6];
    for (int i = 0; i < 6; i++) scanf("%lf", &light_params[i]);
    
    int max_depth, ssaa;
    scanf("%d %d", &max_depth, &ssaa);
    
    const int MAX_TRIGS = 1000;
    trig* scene = (trig*)malloc(sizeof(trig) * MAX_TRIGS);
    int trig_count = 0;
    
    vec3 floor_v[4];
    for (int i = 0; i < 4; i++)
        floor_v[i] = {floor_coords[i*3], floor_coords[i*3+1], floor_coords[i*3+2]};
    
    uchar4 floor_col = make_uchar4((uchar)(floor_color[0]*255), (uchar)(floor_color[1]*255), 
                        (uchar)(floor_color[2]*255), 0);
    scene[trig_count++] = {floor_v[0], floor_v[1], floor_v[2], floor_col};
    scene[trig_count++] = {floor_v[0], floor_v[2], floor_v[3], floor_col};
    
    for (int i = 0; i < 3; i++) {
        vec3 center = {body_params[i][0], body_params[i][1], body_params[i][2]};
        double radius = body_params[i][6];
        uchar4 color = make_uchar4((uchar)(body_params[i][3]*255), (uchar)(body_params[i][4]*255),
                        (uchar)(body_params[i][5]*255), 0);
        
        if (i == 0) add_tetrahedron(scene, trig_count, center, radius, color);
        else if (i == 1) add_hexahedron(scene, trig_count, center, radius, color);
        else add_dodecahedron(scene, trig_count, center, radius, color);
    }
    
    vec3 light_pos = {light_params[0], light_params[1], light_params[2]};
    
    uchar4* data = (uchar4*)malloc(sizeof(uchar4) * w * h);
    trig* d_scene;
    uchar4* d_data;
    
    if (use_gpu) {
        cudaMalloc(&d_scene, sizeof(trig) * trig_count);
        cudaMalloc(&d_data, sizeof(uchar4) * w * h);
        cudaMemcpy(d_scene, scene, sizeof(trig) * trig_count, cudaMemcpyHostToDevice);
    }
    
    for (int frame = 0; frame < frames; frame++) {
        double t = 2.0 * M_PI * frame / frames;
        
        double r_c = cam_params[0] + cam_params[3] * sin(cam_params[5] * t + cam_params[8]);
        double z_c = cam_params[1] + cam_params[4] * sin(cam_params[6] * t + cam_params[9]);
        double phi_c = cam_params[2] + cam_params[7] * t;
        
        double r_n = cam_params[10] + cam_params[13] * sin(cam_params[15] * t + cam_params[18]);
        double z_n = cam_params[11] + cam_params[14] * sin(cam_params[16] * t + cam_params[19]);
        double phi_n = cam_params[12] + cam_params[17] * t;
        
        vec3 pc = {r_c * cos(phi_c), z_c, r_c * sin(phi_c)};
        vec3 pv = {r_n * cos(phi_n), z_n, r_n * sin(phi_n)};
        
        clock_t start = clock();
        
        if (use_gpu) {
            dim3 block(16, 16);
            dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
            render_kernel<<<grid, block>>>(pc, pv, w, h, fov, d_scene, trig_count, light_pos, d_data);
            cudaDeviceSynchronize();
            cudaMemcpy(data, d_data, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost);
        } else {
            render_cpu(pc, pv, w, h, fov, scene, trig_count, light_pos, data);
        }
        
        clock_t end = clock();
        double ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
        
        printf("%d\t%.2f\t%d\n", frame, ms, w * h);
        
        char filename[512];
        sprintf(filename, output_path, frame);
        FILE* fp = fopen(filename, "wb");
        fwrite(&w, sizeof(int), 1, fp);
        fwrite(&h, sizeof(int), 1, fp);
        fwrite(data, sizeof(uchar4), w * h, fp);
        fclose(fp);
    }
    
    if (use_gpu) {
        cudaFree(d_scene);
        cudaFree(d_data);
    }
    free(data);
    free(scene);
    
    return 0;
}