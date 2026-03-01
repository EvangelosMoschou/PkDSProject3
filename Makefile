# =============================================================================
# Project 3 -> Project 4: Parallel BFS using CUDA - Makefile
# V5 Multi-GPU
# =============================================================================

# Compiler
NVCC = /usr/local/cuda/bin/nvcc
CXX = g++

# Build tuning knobs
HOST_NATIVE ?= 1
ENABLE_LTO ?= 0
PGO ?= off
PGO_DIR ?= pgo-data

# Directories
SRC_DIR = src
INC_DIR = include
BIN_DIR = bin
OBJ_DIR = obj

# HDF5 flags (Ported from Project 2)
HDF5_INC = -I/usr/include/hdf5/serial
HDF5_LIB = -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5_serial

# CUDA flags
# NOTE: Added -Xcompiler -fopenmp to support OpenMP local multithreading
NVCC_FLAGS = -std=c++14 -O3 -arch=sm_86 -Xcompiler -fopenmp
NVCC_FLAGS += -I$(INC_DIR) -I$(SRC_DIR)/common $(HDF5_INC)

# Host-side native tuning
ifeq ($(HOST_NATIVE),1)
NVCC_FLAGS += -Xcompiler -march=native -Xcompiler -mtune=native
endif

# Optional host+device LTO
ifeq ($(ENABLE_LTO),1)
NVCC_FLAGS += -dlto
endif

# Profile-Guided Optimization (host compiler)
ifeq ($(PGO),gen)
NVCC_FLAGS += -Xcompiler -fprofile-generate=$(PGO_DIR)
endif
ifeq ($(PGO),use)
NVCC_FLAGS += -Xcompiler -fprofile-use=$(PGO_DIR) -Xcompiler -fprofile-correction
endif

# Debug flags (uncomment for debugging)
# NVCC_FLAGS += -g -G -DDEBUG

# Linker flags
LDFLAGS = -L/usr/local/cuda/lib64 -lcudart $(HDF5_LIB)

ifeq ($(ENABLE_LTO),1)
LDFLAGS += -dlto
endif

ifeq ($(PGO),gen)
LDFLAGS += -Xcompiler -fprofile-generate=$(PGO_DIR)
endif
ifeq ($(PGO),use)
LDFLAGS += -Xcompiler -fprofile-use=$(PGO_DIR) -Xcompiler -fprofile-correction
endif

# Source files
COMMON_SRCS = $(SRC_DIR)/common/graph.cu $(SRC_DIR)/common/utils.cu $(SRC_DIR)/common/json_gpu.cu $(SRC_DIR)/common/io_utils.cu $(SRC_DIR)/common/compression.cu
V5_SRCS = $(SRC_DIR)/main_multi_gpu.cu $(SRC_DIR)/v5_multi_gpu/bfs_multi_gpu.cu $(SRC_DIR)/v5_multi_gpu/bfs_compressed_multi_gpu.cu $(SRC_DIR)/v5_multi_gpu/afforest_multi_gpu.cu $(SRC_DIR)/v5_multi_gpu/bfs_compressed_kernels.cu

# Object files (Common)
COMMON_OBJS = $(OBJ_DIR)/graph.o $(OBJ_DIR)/utils.o $(OBJ_DIR)/json_gpu.o $(OBJ_DIR)/io_utils.o $(OBJ_DIR)/compression.o

# Object files (V5)
V5_OBJS = $(OBJ_DIR)/main_multi_gpu.o $(OBJ_DIR)/bfs_multi_gpu.o $(OBJ_DIR)/bfs_compressed_multi_gpu.o $(OBJ_DIR)/afforest_multi_gpu.o $(OBJ_DIR)/bfs_compressed_kernels.o

# Object files (V4.1)
V41_OBJS = $(OBJ_DIR)/main_v41.o $(OBJ_DIR)/bfs_adaptive_v41.o $(OBJ_DIR)/bfs_compressed_adaptive_v41.o $(OBJ_DIR)/bfs_compressed_kernels_v41.o $(OBJ_DIR)/afforest_v41.o

# Executables
V5_BIN = $(BIN_DIR)/bfs_v5_multi_gpu
V41_BIN = $(BIN_DIR)/bfs_v4_1_hybrid

# =============================================================================
# Targets
# =============================================================================

.PHONY: all clean dirs v41 mat_to_csrbin_tool pgo-gen pgo-use

all: dirs $(V5_BIN)

v41: dirs $(V41_BIN)

dirs:
	@mkdir -p $(BIN_DIR) $(OBJ_DIR)

# Version 5: Multi-GPU Development Build
$(V5_BIN): $(COMMON_OBJS) $(V5_OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)

$(OBJ_DIR)/main_multi_gpu.o: $(SRC_DIR)/main_multi_gpu.cu
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

$(OBJ_DIR)/bfs_multi_gpu.o: $(SRC_DIR)/v5_multi_gpu/bfs_multi_gpu.cu
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

$(OBJ_DIR)/bfs_compressed_multi_gpu.o: $(SRC_DIR)/v5_multi_gpu/bfs_compressed_multi_gpu.cu
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

$(OBJ_DIR)/bfs_compressed_kernels.o: $(SRC_DIR)/v5_multi_gpu/bfs_compressed_kernels.cu
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

$(OBJ_DIR)/afforest_multi_gpu.o: $(SRC_DIR)/v5_multi_gpu/afforest_multi_gpu.cu
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

# Version 4.1: Hybrid Build
$(V41_BIN): $(COMMON_OBJS) $(V41_OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)

$(OBJ_DIR)/main_v41.o: $(SRC_DIR)/main_multi_gpu.cu
	$(NVCC) $(NVCC_FLAGS) -DUSE_V41_HYBRID -c -o $@ $<

$(OBJ_DIR)/bfs_adaptive_v41.o: $(SRC_DIR)/v4_1_hybrid/bfs_adaptive.cu
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

$(OBJ_DIR)/bfs_compressed_adaptive_v41.o: $(SRC_DIR)/v4_1_hybrid/bfs_compressed_adaptive.cu
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

$(OBJ_DIR)/bfs_compressed_kernels_v41.o: $(SRC_DIR)/v4_1_hybrid/bfs_compressed_kernels.cu
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

$(OBJ_DIR)/afforest_v41.o: $(SRC_DIR)/v4_1_hybrid/afforest.cu
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
reorder_graph: dirs $(COMMON_OBJS) $(OBJ_DIR)/reorder.o $(OBJ_DIR)/reorder_main.o
	$(NVCC) $(NVCC_FLAGS) -o $(BIN_DIR)/$@ $(COMMON_OBJS) $(OBJ_DIR)/reorder.o $(OBJ_DIR)/reorder_main.o $(LDFLAGS)

$(OBJ_DIR)/reorder_main.o: $(SRC_DIR)/reorder_main.cu
	$(NVCC) $(NVCC_FLAGS) -I$(SRC_DIR)/common -c -o $@ $<

$(OBJ_DIR)/mat_to_csrbin_tool.o: $(SRC_DIR)/tools/mat_to_csrbin.cu
	$(NVCC) $(NVCC_FLAGS) -I$(SRC_DIR) -I$(SRC_DIR)/common -c -o $@ $<

mat_to_csrbin_tool: dirs $(COMMON_OBJS) $(OBJ_DIR)/mat_to_csrbin_tool.o
	$(NVCC) $(NVCC_FLAGS) -o $(BIN_DIR)/mat_to_csrbin $(COMMON_OBJS) $(OBJ_DIR)/mat_to_csrbin_tool.o $(LDFLAGS)

# PGO helper targets:
# 1) make pgo-gen
# 2) Run representative workload(s) to generate profiles
# 3) make pgo-use
pgo-gen: dirs
	@mkdir -p $(PGO_DIR)
	$(MAKE) clean
	$(MAKE) v41 HOST_NATIVE=$(HOST_NATIVE) ENABLE_LTO=$(ENABLE_LTO) PGO=gen PGO_DIR=$(PGO_DIR)
	@echo ""
	@echo "PGO generation build completed."
	@echo "Run training workload, then execute: make pgo-use"

pgo-use: dirs
	$(MAKE) clean
	$(MAKE) v41 HOST_NATIVE=$(HOST_NATIVE) ENABLE_LTO=$(ENABLE_LTO) PGO=use PGO_DIR=$(PGO_DIR)

# Clean
clean:
	rm -rf $(BIN_DIR) $(OBJ_DIR)

# Help
help:
	@echo "Available targets:"
	@echo "  all          - Build Multi-GPU version (v5)"
	@echo "  v41          - Build V4.1 hybrid solver"
	@echo "  reorder_graph - Build graph reordering tool"
	@echo "  mat_to_csrbin_tool - Build MAT->CSRBIN converter"
	@echo "  pgo-gen      - Build instrumented binary for profile collection"
	@echo "  pgo-use      - Build using collected profile data"
	@echo "  clean        - Remove build files"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Build options (override via make VAR=value):"
	@echo "  HOST_NATIVE=1|0   (default: 1)"
	@echo "  ENABLE_LTO=1|0    (default: 0)"
	@echo "  PGO=off|gen|use   (default: off)"
	@echo "  PGO_DIR=<path>    (default: pgo-data)"
	@echo ""
