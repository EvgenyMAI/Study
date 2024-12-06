#include "lib.h"
#include <stdio.h>

int main() {
    int command;

    while (scanf("%d", &command) != EOF) {
        if (command == 1) {
            int first_number, second_number;
            printf("Enter 2 numbers for searching GCF: ");
            scanf("%d %d", &first_number, &second_number);
            printf("The result of the first function: %d\n", GCF(first_number, second_number));
        } else if (command == 2) {
            float length, width;
            printf("Enter the length and width of the shape: ");
            scanf("%f %f", &length, &width);
            printf("The area of the figure: %f\n", Square(length, width));
        } else if (command == -1) {
            printf("The program is completed.\n");
            break;
        } else {
            printf("You entered the wrong command.\n");
        }
    }

    return 0;
}