CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra
TARGET = tokenizer
TEST_TARGET = test_tokenizer

# Исходные файлы
SRC = tokenizer.cpp main.cpp
TEST_SRC = tokenizer.cpp test_tokenizer.cpp
OBJ = $(SRC:.cpp=.o)
TEST_OBJ = $(TEST_SRC:.cpp=.o)

# Основная цель
all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Тесты
test: $(TEST_TARGET)
	./$(TEST_TARGET)

$(TEST_TARGET): $(TEST_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Компиляция объектных файлов
%.o: %.cpp tokenizer.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Очистка
clean:
	rm -f $(OBJ) $(TEST_OBJ) $(TARGET) $(TEST_TARGET)
	rm -f *.exe *.o

.PHONY: all test clean