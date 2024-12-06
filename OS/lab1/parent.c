#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <sys/types.h>


const int BUFFER_SIZE = 1000;


int main() {
    char first_filename[BUFFER_SIZE], second_filename[BUFFER_SIZE];


    printf("Введите имя файла для первого дочернего процесса: ");
    fgets(first_filename, BUFFER_SIZE, stdin);
    first_filename[strcspn(first_filename, "\n")] = 0;


    printf("Введите имя файла для второго дочернего процесса: ");
    fgets(second_filename, BUFFER_SIZE, stdin);
    second_filename[strcspn(second_filename, "\n")] = 0;


    int first_file_descriptor = open(first_filename, O_CREAT | O_WRONLY, S_IRWXU);
    int second_file_descriptor = open(second_filename, O_CREAT | O_WRONLY, S_IRWXU);


    if (first_file_descriptor < 0 || second_file_descriptor < 0) {
        perror("Ошибка открытия файла(ов)");
        exit(EXIT_FAILURE);
    }


    int first_pipe[2], second_pipe[2];
    if (pipe(first_pipe) < 0 || pipe(second_pipe) < 0) {
        perror("Ошибка создания канала(ов) связи");
        exit(EXIT_FAILURE);
    }


    // Дочерние процессы


    pid_t first_process_id = fork();
    if (first_process_id < 0) {
        perror("Ошибка создания первого дочернего процесса");
        exit(EXIT_FAILURE);
    }
    if (first_process_id == 0) {
        close(first_pipe[1]);
        close(second_pipe[1]);
        close(second_pipe[0]);


        dup2(first_pipe[0], STDIN_FILENO);
        dup2(first_file_descriptor, STDOUT_FILENO);


        if (execl("./child", "child", NULL) < 0) {
            perror("Ошибка вызова execl для первого дочернего процесса");
            exit(EXIT_FAILURE);
        }
    }


    pid_t second_process_id = fork();
    if (second_process_id < 0) {
        perror("Ошибка создания второго дочернего процесса");
        exit(EXIT_FAILURE);
    }
    if (second_process_id == 0) {
        close(second_pipe[1]);
        close(first_pipe[1]);
        close(first_pipe[0]);


        dup2(second_pipe[0], STDIN_FILENO);
        dup2(second_file_descriptor, STDOUT_FILENO);


        if (execl("./child", "child", NULL) < 0) {
            perror("Ошибка вызова execl для второго дочернего процесса");
            exit(EXIT_FAILURE);
        }
    }


    // Родительский процесс


    close(first_pipe[0]);
    close(second_pipe[0]);


    char buffer[BUFFER_SIZE];
    int string_parity = 1;


    printf("Введите строки:\n");
    while(fgets(buffer, BUFFER_SIZE, stdin) != NULL) {
        int len_buffer = strlen(buffer);
        if (string_parity % 2 == 0)
            write(second_pipe[1], buffer, len_buffer);
        else 
            write(first_pipe[1], buffer, len_buffer);

        ++string_parity;
    }

    close(first_pipe[1]);
    close(second_pipe[1]);


    waitpid(first_process_id, NULL, 0);
    waitpid(second_process_id, NULL, 0);


    return 0;
}