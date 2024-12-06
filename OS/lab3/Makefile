CC = gcc
CFLAGS = -Wall

all: parent child

parent: parent.c
	$(CC) $(CFLAGS) -o parent parent.c -lrt -lpthread

child: child.c
	$(CC) $(CFLAGS) -o child child.c -lrt -lpthread