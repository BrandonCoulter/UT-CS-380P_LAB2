# Paths
SRC_DIR := ./src
BIN_DIR := ./bin
BUILD_DIR := ./build
INC_DIR := ./src

# Create directories if they don't exist already
$(shell mkdir -p $(BIN_DIR) $(BUILD_DIR))

# Executable file name
EXE := $(BIN_DIR)/kmeans

# Collect all .cpp and .cu files and create corresponding .o files in build directory
CPP_SRC := $(wildcard $(SRC_DIR)/*.cpp)
CU_SRC := $(wildcard $(SRC_DIR)/*.cu)
OBJ := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(CPP_SRC)) \
       $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CU_SRC))

# Compiler and flags
NVCC := nvcc

# Optimization Flags
OPT_FLAGS := -O3

# Compiler flags for host code (C++ files)
CXXFLAGS := -I$(INC_DIR) $(OPT_FLAGS) -Xcompiler -Wall 

# Flags for CUDA code (device code)
CUFLAGS := -arch=sm_75 $(OPT_FLAGS)

# Target to build the executable
all: $(EXE)

# Rule to create the executable (linking object files from build directory)
$(EXE): $(OBJ)
	$(NVCC) -o $@ $^

# Rule to compile .cpp files to .o in build directory
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(NVCC) -c $(CXXFLAGS) $< -o $@

# Rule to compile .cu files to .o in build directory
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) -c $(CUFLAGS) $< -o $@

# Clean up object files and executable
clean:
	rm -rf $(BUILD_DIR) $(EXE)
