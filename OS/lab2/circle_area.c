#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <math.h>
#include <time.h>


const int ALL_POINTS = 1e7;
const double PI = 3.141592;


typedef struct {
    double radius;
    int number_points;
    double result;
} Thread_Data;


double random_in_range(double min, double max) {
    return min + (max - min) * ((double) rand() / RAND_MAX);
}


void* calculateArea(void* arg) {
    Thread_Data* data = (Thread_Data*) arg;
    double radius = data->radius;
    int number_points = data->number_points;


    int points_inside_circle = 0;


    for (int i = 0; i < number_points; ++i) {
        double x = random_in_range(-radius, radius);
        double y = random_in_range(-radius, radius);

        if (sqrt(x * x + y * y) <= radius)
            ++points_inside_circle;
    }


    // Расчет площади окружности в текущем потоке
    double area = (double)points_inside_circle  / (double)number_points * 4 * radius * radius;


    data->result = area;


    pthread_exit(NULL);
}


int main(int argc, char* argv[]) {
    clock_t start_time, end_time;
    start_time = clock();


    if (argc != 3) {
        printf("Incorrect count of arguments\n");
        return 1;
    }


    double radius = atof(argv[1]);
    int number_threads = atoi(argv[2]);


    if (radius <= 0 || number_threads <= 0) {
        printf("Invalid arguments\n");
        return 1;
    }


    pthread_t* threads = (pthread_t*) malloc(number_threads * sizeof(pthread_t));
    Thread_Data* threadData = (Thread_Data*) malloc(number_threads * sizeof(Thread_Data));


    int number_points_per_thread = ALL_POINTS / number_threads;


    for (int i = 0; i < number_threads; ++i) {
        threadData[i].radius = radius;
        threadData[i].number_points = number_points_per_thread;
        
        pthread_create(&threads[i], NULL, calculateArea, (void*) &threadData[i]);
    }


    for (int i = 0; i < number_threads; ++i) {
        pthread_join(threads[i], NULL);
    }


    double sum_areas = 0.0;
    for (int i = 0; i < number_threads; ++i) {
        sum_areas += threadData[i].result;
    }


    double circle_area = sum_areas / number_threads;
    double reference = PI * radius * radius;


    end_time = clock();
    double cpu_time_used = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;


    printf("Time is %.2f seconds\n", cpu_time_used);
    printf("Radius = %.2f\n", radius);
    printf("Reference Аrea of the circle (S=Pi*r^2) = %.2f\n", reference);
    printf("Аrea of the circle calculating with the Monte-Carlo method = %.2f\n", circle_area);


    free(threads);
    free(threadData);


    return 0;
}