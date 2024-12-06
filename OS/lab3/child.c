#include "stdio.h"
#include "fcntl.h"
#include "stdlib.h"
#include "unistd.h"
#include "string.h"
#include "sys/wait.h"
#include "sys/stat.h"
#include "sys/mman.h"
#include "sys/types.h"
#include "semaphore.h"

const int BUFFER_SIZE = 1000;
const char vowels[] = {'a', 'e', 'i', 'o', 'u', 'y', 'A', 'E', 'I', 'O', 'U', 'Y'};

int main(int argc, char* argv[]) {
    char file_name[BUFFER_SIZE];
    char mmapped_file_name[BUFFER_SIZE];
    strcpy(file_name, argv[1]);
    strcpy(mmapped_file_name, argv[2]);

    char semaphore_name[BUFFER_SIZE];
    char semaphore_for_parent_name[BUFFER_SIZE];
    strcpy(semaphore_name, argv[3]);
    strcpy(semaphore_for_parent_name, argv[4]);

    int file_descriptor = open(file_name, O_RDWR | O_CREAT | O_TRUNC, S_IRWXU);

    int mmapped_file_descriptor = shm_open(mmapped_file_name, O_RDWR | O_CREAT, 0666);
    ftruncate(mmapped_file_descriptor, BUFFER_SIZE);
    char* mmapped_file_pointer = mmap(NULL, BUFFER_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, mmapped_file_descriptor, 0);

    sem_t* semaphore = sem_open(semaphore_name, 0);
    sem_t* semaphore_for_parent = sem_open(semaphore_for_parent_name, 0);

    char string[BUFFER_SIZE];
    while(1) {
        sem_wait(semaphore);
        size_t string_length = strlen(mmapped_file_pointer);
        for (int i = 0; i < string_length; ++i)
            string[i] = mmapped_file_pointer[i];
        string[string_length] = 0;

        sem_post(semaphore_for_parent);

        if (string_length == 0) 
            break;

        for (int index = 0; index < strlen(string); ++index) {
            if (memchr(vowels, string[index], 12) == NULL)
                write(file_descriptor, &string[index], 1);
        }
    }

    munmap(mmapped_file_pointer, 0);
    sem_close(semaphore);
    sem_close(semaphore_for_parent);
    close(file_descriptor);

    return 0;
}