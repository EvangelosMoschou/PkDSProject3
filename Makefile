# =============================================================================
# Project 3: Parallel BFS using CUDA - Makefile
# =============================================================================

# Compiler
NVCC = nvcc
CXX = g++

# Directories
SRC_DIR = src
INC_DIR = include
BIN_DIR = bin
OBJ_DIR = obj

# CUDA flags
NVCC_FLAGS = -std=c++14 -O3 -arch=sm_86
NVCC_FLAGS += -I$(INC_DIR) -I$(SRC_DIR)/common

# Debug flags (uncomment for debugging)
# NVCC_FLAGS += -g -G -DDEBUG

# Linker flags
LDFLAGS = -lcudart

# Source files
COMMON_SRCS = $(SRC_DIR)/common/graph.cu $(SRC_DIR)/common/utils.cu
V1_SRCS = $(SRC_DIR)/v1_dynamic/bfs_dynamic.cu
V2_SRCS = $(SRC_DIR)/v2_chunked/bfs_chunked.cu
V3_SRCS = $(SRC_DIR)/v3_shared/bfs_shared.cu

# Object files
COMMON_OBJS = $(OBJ_DIR)/graph.o $(OBJ_DIR)/utils.o
V1_OBJS = $(OBJ_DIR)/bfs_dynamic.o
V2_OBJS = $(OBJ_DIR)/bfs_chunked.o
V3_OBJS = $(OBJ_DIR)/bfs_shared.o

# Executables
V1_BIN = $(BIN_DIR)/bfs_v1
V2_BIN = $(BIN_DIR)/bfs_v2
V3_BIN = $(BIN_DIR)/bfs_v3

# =============================================================================
# Targets
# =============================================================================

.PHONY: all v1 v2 v3 clean dirs

all: dirs v1 v2 v3

dirs:
	@mkdir -p $(BIN_DIR) $(OBJ_DIR)

# Version 1: Dynamic Thread Assignment
v1: dirs $(V1_BIN)

$(V1_BIN): $(COMMON_OBJS) $(V1_OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)

$(OBJ_DIR)/bfs_dynamic.o: $(V1_SRCS)
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

# Version 2: Chunk-Based Processing
v2: dirs $(V2_BIN)

$(V2_BIN): $(COMMON_OBJS) $(V2_OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)

$(OBJ_DIR)/bfs_chunked.o: $(V2_SRCS)
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

# Version 3: Shared Memory with Warp Cooperation
v3: dirs $(V3_BIN)

$(V3_BIN): $(COMMON_OBJS) $(V3_OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)

$(OBJ_DIR)/bfs_shared.o: $(V3_SRCS)
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

# Common objects
$(OBJ_DIR)/graph.o: $(SRC_DIR)/common/graph.cu
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

$(OBJ_DIR)/utils.o: $(SRC_DIR)/common/utils.cu
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

# Clean
clean:
	rm -rf $(BIN_DIR) $(OBJ_DIR)

# Help
help:
	@echo "Available targets:"
	@echo "  all     - Build all versions"
	@echo "  v1      - Build Version 1 (Dynamic Thread Assignment)"
	@echo "  v2      - Build Version 2 (Chunk-Based Processing)"
	@echo "  v3      - Build Version 3 (Shared Memory + Warp Cooperation)"
	@echo "  clean   - Remove build files"
	@echo "  help    - Show this help message"
