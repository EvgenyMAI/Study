CC = gcc
CFLAGS = -Wall -Wextra
LDFLAGS = -ldl

SRC_DIR = .
BUILD_DIR = trash

SRC_FILES = dynamic_lib.c lib_first.c lib_second.c
OBJ_FILES = $(patsubst %.c, $(BUILD_DIR)/%.o, $(SRC_FILES))
LIB_FILES = $(patsubst %.c, $(BUILD_DIR)/lib%.so, $(filter-out dynamic_lib.c, $(SRC_FILES)))

.PHONY: all clean

all: dynamic_lib static_lib_first static_lib_second

dynamic_lib: $(OBJ_FILES) $(LIB_FILES)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/lib%.so: $(SRC_DIR)/%.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -fPIC -shared $< -o $@

static_lib_first: $(BUILD_DIR)/static_lib_first.o $(BUILD_DIR)/lib_first.a
	$(CC) $(CFLAGS) $^ -o $@

$(BUILD_DIR)/static_lib_first.o: static_lib.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/lib_first.a: $(BUILD_DIR)/lib_first.o
	ar rcs $@ $<

$(BUILD_DIR)/lib_first.o: lib_first.c
	$(CC) $(CFLAGS) -c $< -o $@

static_lib_second: $(BUILD_DIR)/static_lib_second.o $(BUILD_DIR)/lib_second.a
	$(CC) $(CFLAGS) $^ -o $@

$(BUILD_DIR)/static_lib_second.o: static_lib.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/lib_second.a: $(BUILD_DIR)/lib_second.o
	ar rcs $@ $<

$(BUILD_DIR)/lib_second.o: lib_second.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) dynamic_lib static_lib_first static_lib_second
