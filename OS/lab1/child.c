#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>


const int BUFFER_SIZE = 1000;


int is_vowel(char symbol) {
    switch(symbol) {
        case 'a':
        case 'e':
        case 'i':
        case 'o':
        case 'u':
        case 'A':
        case 'E':
        case 'I':
        case 'O':
        case 'U':
            return 1;

        default:
            return 0;
    }
}


void remove_vowels(char* string, char* new_string) {
    int count_string, count_new_string;
    for (count_string = 0, count_new_string = 0; count_string < strlen(string); ++count_string) {
        if (!is_vowel(string[count_string]))
            new_string[count_new_string++] = string[count_string];
    }

    new_string[count_new_string] = '\0';
}


int main() {
    char string[BUFFER_SIZE];
    char new_string[BUFFER_SIZE];


    while (fgets(string, BUFFER_SIZE, stdin) != NULL) {
        remove_vowels(string, new_string);
        write(STDOUT_FILENO, new_string, strlen(new_string));
    }


    close(STDOUT_FILENO);
    close(STDIN_FILENO);
    return 0;
}