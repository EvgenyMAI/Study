#include "stdio.h"
#include "fcntl.h"
#include "string.h"
#include "stdlib.h"
#include "unistd.h"
#include "sys/mman.h"
#include "sys/wait.h"
#include "sys/stat.h"
#include "sys/types.h"
#include "semaphore.h"

const int BUFFER_SIZE = 1000;

int main() {
    char first_filename[BUFFER_SIZE], second_filename[BUFFER_SIZE];

    printf("Введите имя файла для первого дочернего процесса: ");
    fgets(first_filename, BUFFER_SIZE, stdin);
    first_filename[strcspn(first_filename, "\n")] = 0;

    printf("Введите имя файла для второго дочернего процесса: ");
    fgets(second_filename, BUFFER_SIZE, stdin);
    second_filename[strcspn(second_filename, "\n")] = 0;

    int mmapped_file_1_descriptor = shm_open("mmaped_file_1", O_RDWR | O_CREAT, 0666);
    int mmapped_file_2_descriptor = shm_open("mmaped_file_2", O_RDWR | O_CREAT, 0666);

    ftruncate(mmapped_file_1_descriptor, BUFFER_SIZE);
    ftruncate(mmapped_file_2_descriptor, BUFFER_SIZE);

    char* mmapped_file_1_pointer = mmap(NULL, BUFFER_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, mmapped_file_1_descriptor, 0); //отображение файла в адресное пространство процесса
    char* mmapped_file_2_pointer = mmap(NULL, BUFFER_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, mmapped_file_2_descriptor, 0);

    sem_t* semaphore_one = sem_open("/semaphoreOne", O_CREAT, 0666, 0);
    sem_t* semaphore_two = sem_open("/semaphoreTwo", O_CREAT, 0666, 0);

    sem_t* semaphore_for_parent_one = sem_open("/semaphoresForParentOne", O_CREAT, 0666, 1);
    sem_t* semaphore_for_parent_two = sem_open("/semaphoresForParentTwo", O_CREAT, 0666, 1);

    pid_t process_id_1 = fork();
    if (process_id_1 < 0) {
        perror("Ошибка создания первого дочернего процесса");
        exit(EXIT_FAILURE);
    }
    if (process_id_1 == 0) {
        execl("child", "", first_filename, "mmaped_file_1", "/semaphoreOne", "/semaphoresForParentOne", NULL);
        perror("Ошибка вызова функции execl() для первого дочернего процесса");
        exit(EXIT_FAILURE);
    }

    pid_t process_id_2 = fork();
    if (process_id_2 < 0) {
        perror("Ошибка создания второго дочернего процесса");
        exit(EXIT_FAILURE);
    }
    if (process_id_2 == 0) {
        execl("child", "", second_filename, "mmaped_file_2", "/semaphoreTwo", "/semaphoresForParentTwo", NULL);
        perror("Ошибка вызова функции execl() для второго дочернего процесса");
        exit(EXIT_FAILURE);
    }

    char string[BUFFER_SIZE];
    int counter = 0;
    while (fgets(string, BUFFER_SIZE, stdin)) {
        size_t len = strlen(string);
        if (++counter % 2 == 1) {
            sem_wait(semaphore_for_parent_one);
            memcpy(mmapped_file_1_pointer, string, len + 1); // включая нуль-терминатор
            sem_post(semaphore_one);
        } else {
            sem_wait(semaphore_for_parent_two);
            memcpy(mmapped_file_2_pointer, string, len + 1); // включая нуль-терминатор
            sem_post(semaphore_two);
        }
    }
    
    sem_wait(semaphore_for_parent_one);
    sem_wait(semaphore_for_parent_two);
    mmapped_file_1_pointer[0] = 0;
    mmapped_file_2_pointer[0] = 0;
    sem_post(semaphore_one);
    sem_post(semaphore_two);

    wait(NULL);
    wait(NULL);

    munmap(mmapped_file_1_pointer, BUFFER_SIZE);
    munmap(mmapped_file_2_pointer, BUFFER_SIZE);

    shm_unlink("mmaped_file_1");
    shm_unlink("mmaped_file_2");

    sem_close(semaphore_one);
    sem_close(semaphore_two);

    sem_close(semaphore_for_parent_one);
    sem_close(semaphore_for_parent_two);

    return 0;
}