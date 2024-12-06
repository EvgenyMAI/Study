SRC = main.cpp tree.cpp
OBJ = $(SRC:.cpp=.o)

SRC2 = client.cpp
OBJ2 = $(SRC2:.cpp=.o)

all: main client

main: $(OBJ)
	g++ -Wno-unused-variable $(OBJ) -o $@ -lrt -lzmq

client: $(OBJ2)
	g++ -Wno-unused-variable $(OBJ2) -o $@ -lrt -lzmq

.cpp.o:
	g++ -Wno-unused-variable -c $< -o $@

client.o: tree.h 
main.o: tree.h
tree.o: tree.h

clean:
	rm client.o main.o  tree.o main client