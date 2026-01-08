# =============================================================================
# Project 3: Parallel BFS using CUDA - Makefile
# =============================================================================

# Compiler
NVCC = /usr/local/cuda/bin/nvcc
CXX = g++

# Directories
SRC_DIR = src
INC_DIR = include
BIN_DIR = bin
OBJ_DIR = obj

# HDF5 flags (Ported from Project 2)
HDF5_INC = -I/usr/include/hdf5/serial
HDF5_LIB = -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5_serial

# CUDA flags
NVCC_FLAGS = -std=c++14 -O3 -arch=sm_86
NVCC_FLAGS += -I$(INC_DIR) -I$(SRC_DIR)/common $(HDF5_INC)

# Debug flags (uncomment for debugging)
# NVCC_FLAGS += -g -G -DDEBUG

# Linker flags
LDFLAGS = -L/usr/local/cuda/lib64 -lcudart $(HDF5_LIB)

# Source files
COMMON_SRCS = $(SRC_DIR)/common/graph.cu $(SRC_DIR)/common/utils.cu $(SRC_DIR)/common/json_gpu.cu $(SRC_DIR)/common/io_utils.cu $(SRC_DIR)/common/compression.cu
V3_SRCS = $(SRC_DIR)/legacy/v3_shared/bfs_shared.cu $(SRC_DIR)/v4_adaptive/bfs_adaptive.cu

# Object files
COMMON_OBJS = $(OBJ_DIR)/graph.o $(OBJ_DIR)/utils.o $(OBJ_DIR)/json_gpu.o $(OBJ_DIR)/io_utils.o $(OBJ_DIR)/compression.o
V3_OBJS = $(OBJ_DIR)/bfs_shared.o $(OBJ_DIR)/bfs_adaptive.o \
          $(OBJ_DIR)/bfs_compressed_kernels.o $(OBJ_DIR)/bfs_compressed_adaptive.o \
          $(OBJ_DIR)/afforest.o

# Executables
V3_BIN = $(BIN_DIR)/bfs_v3

# =============================================================================
# Targets
# =============================================================================

.PHONY: all v3 clean dirs

all: dirs v3

dirs:
	@mkdir -p $(BIN_DIR) $(OBJ_DIR)

# Version 3: Production Build (Adaptive BFS + Compressed BFS + Afforest)
v3: dirs $(V3_BIN)

$(V3_BIN): $(COMMON_OBJS) $(V3_OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)

$(OBJ_DIR)/bfs_shared.o: $(SRC_DIR)/legacy/v3_shared/bfs_shared.cu
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

$(OBJ_DIR)/bfs_adaptive.o: $(SRC_DIR)/v4_1_hybrid/bfs_adaptive.cu
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

$(OBJ_DIR)/bfs_compressed_kernels.o: $(SRC_DIR)/legacy/v3_shared/bfs_compressed_kernels.cu
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

$(OBJ_DIR)/bfs_compressed_adaptive.o: $(SRC_DIR)/v4_1_hybrid/bfs_compressed_adaptive.cu
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

$(OBJ_DIR)/afforest.o: $(SRC_DIR)/v4_1_hybrid/afforest.cu
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

# Common objects
$(OBJ_DIR)/graph.o: $(SRC_DIR)/common/graph.cu
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

$(OBJ_DIR)/utils.o: $(SRC_DIR)/common/utils.cu
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

$(OBJ_DIR)/json_gpu.o: $(SRC_DIR)/common/json_gpu.cu
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

$(OBJ_DIR)/io_utils.o: $(SRC_DIR)/common/io_utils.cu
	$(NVCC) $(NVCC_FLAGS) -x cu -c -o $@ $<

$(OBJ_DIR)/compression.o: $(SRC_DIR)/common/compression.cu
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

$(OBJ_DIR)/reorder.o: $(SRC_DIR)/common/reorder.cu
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

# Tools
reorder_graph: $(COMMON_OBJS) $(OBJ_DIR)/reorder.o $(OBJ_DIR)/reorder_main.o
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)

$(OBJ_DIR)/reorder_main.o: $(SRC_DIR)/reorder_main.cu
	$(NVCC) $(NVCC_FLAGS) -I$(SRC_DIR)/common -c -o $@ $<

# Clean
clean:
	rm -rf $(BIN_DIR) $(OBJ_DIR)

# Help
help:
	@echo "Available targets:"
	@echo "  all     - Build production version (v3)"
	@echo "  v3      - Build v3 (Adaptive BFS + Compressed BFS + Afforest)"
	@echo "  clean   - Remove build files"
	@echo "  help    - Show this help message"
	@echo ""
	@echo "Legacy versions (v1, v2) are archived in src/legacy/"
