#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <dlfcn.h>

const char* FIRST_LIBRARY_PATH = "trash/liblib_first.so";
const char* SECOND_LIBRARY_PATH = "trash/liblib_second.so";

void* library_descriptor = NULL;
int current_library = 1;

int (* GCF_func)(int A, int B) = NULL;
float (* Square_func)(float A, float B) = NULL;

// Обработка ошибок
void error_processing(bool exception, char* bug_report) {
    if (exception) {
        fprintf(stderr, "%s", bug_report);
        exit(-1);
    }
}

void open_library(const char* path_to_library) {
    if (library_descriptor != NULL) { // Закрываем открытую библиотеку
        dlclose(library_descriptor);
    }

    library_descriptor = dlopen(path_to_library, RTLD_LAZY);
    error_processing(library_descriptor == NULL, "Library opening error\n");

    // Указатели на функции
    GCF_func = dlsym(library_descriptor, "GCF");
    error_processing(GCF_func == NULL, "Error in finding the method GCF\n");

    Square_func = dlsym(library_descriptor, "Square");
    error_processing(Square_func == NULL, "Error in finding the method Square\n");
}

void swap_library() {
    if (current_library == 1) {
        open_library(SECOND_LIBRARY_PATH);
        current_library = 2;
    } else {
        open_library(FIRST_LIBRARY_PATH);
        current_library = 1;
    }

    printf("Current library is %d\n", current_library);
}

int main() {
    open_library(FIRST_LIBRARY_PATH);
    int command;

    while (scanf("%d", &command) != EOF) {
        if (command == 0) {
            swap_library();
        } else if (command == 1) {
            int first_number, second_number;
            printf("Enter 2 numbers for searching GCF: ");
            scanf("%d %d", &first_number, &second_number);
            printf("The result of the first function: %d\n", (*GCF_func)(first_number, second_number));
        } else if (command == 2) {
           float length, width;
            printf("Enter the length and width of the shape: ");
            scanf("%f %f", &length, &width);
            printf("The area of the figure: %f\n", (*Square_func)(length, width));
        } else if (command == -1) {
            printf("The program is completed.\n");
            break;
        } else
            printf("You entered the wrong command.\n");
    }

    dlclose(library_descriptor);
    return 0;
}