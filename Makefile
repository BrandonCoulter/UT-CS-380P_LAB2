# CC = g++ 
# SRCS = ./src/*.cpp
# INC = ./src/
# NEW_OPTS = -std=c++17 -Wall -Werror -O3
# OPTS = -std=c++17 -Wall -Werror -lpthread -O3
# DOPTS = -std=c++17 -Wall -Werror -lpthread -O0 -g -fsanitize=address
# PRINT = -D__PRINT__
# VERBOSE = -D__VERBOSE__

# EXEC = bin/kmeans

# all: clean compile
# print: clean pcompile
# debug: clean dcompile

# compile:
# 	$(CC) $(SRCS) $(OPTS) -I$(INC) -o $(EXEC)

# pcompile:
# 	$(CC) $(SRCS) $(OPTS) ${PRINT} -I$(INC) -o $(EXEC)

# dcompile:
# 	$(CC) $(SRCS) $(DOPTS) ${PRINT} ${VERBOSE} -I$(INC) -o $(EXEC)

# clean:
# 	rm -f $(EXEC)


# Paths
SRC_DIR := ./src
BIN_DIR := ./bin
INC_DIR := ./src

# Create bin folder if it doesn't exist
$(shell mkdir -p $(BIN_DIR))

# Executable file name
EXE := $(BIN_DIR)/kmeans

# Collect all .cpp files and create corresponding .o files
SRC := $(wildcard $(SRC_DIR)/*.cpp)
OBJ := $(patsubst $(SRC_DIR)/%.cpp,$(SRC_DIR)/%.o,$(SRC))

# Compiler and flags
NVCC := nvcc
CXXFLAGS := -I$(INC_DIR)

# Target to build the executable
all: $(EXE)

# Rule to create the executable
$(EXE): $(OBJ)
	$(NVCC) -o $@ $^

# Rule to compile .cpp files to .o
$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp $(SRC_DIR)/%.h
	$(NVCC) -c $(CXXFLAGS) $< -o $@

# Clean up object files and executable
clean:
	rm -f $(OBJ) $(EXE)
