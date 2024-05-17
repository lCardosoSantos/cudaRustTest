## Copyright 2022-2024 Dag Arne Osvik
## Copyright 2022-2024 Luan Cardoso dos Santos

# Compiler
NVCC := nvcc #-g -G -O0

# Compilation flags
CUDAFLAGS := -Iinclude -Iinclude/ptx -Itests/include -rdc=true --std=c++14 -Xlinker=--no-relax --gpu-architecture=compute_86 --gpu-code=sm_86 

# Define statements
DEFINES :=

# Directories
SRCDIR := src
INCDIR := include
TESTDIR := tests
BUILDDIR := build

# Excluded source files
EXCLUDES := ./$(TESTDIR)/test_fp.cu ./$(TESTDIR)/test_fr.cu ./$(TESTDIR)/test_SpMV.cu

# Auto-detect all CUDA source files in src and tests directories, excluding any specified
SRC := $(filter-out $(EXCLUDES), $(shell find ./ -type f -name '*.cu'))

# Create a list of object files by replacing the source directory with the build directory in the path, and replacing .cu with .o
OBJS := $(SRC:%.cu=$(BUILDDIR)/%.o)

# Default rule builds all tests
all: $(BUILDDIR)/test_fp $(BUILDDIR)/test_fr $(BUILDDIR)/test_SpMV

SpMV: $(BUILDDIR)/test_SpMV
	./$<

fp: $(BUILDDIR)/test_fp
	./$<

fr: $(BUILDDIR)/test_fr
	./$<

# Compile CUDA source files to object files
$(BUILDDIR)/%.o: %.cu
	@mkdir -p $(@D)
	$(NVCC) $(DEFINES) $(CUDAFLAGS) --compile -c $< -o $@

# Rules for tests
$(BUILDDIR)/test_fp: $(BUILDDIR)/$(TESTDIR)/test_fp.o $(filter-out $(BUILDDIR)/$(TESTDIR)/test_fp.o, $(OBJS))
	$(NVCC) $(DEFINES) $(CUDAFLAGS) $^ -o $@

$(BUILDDIR)/test_fr: $(BUILDDIR)/$(TESTDIR)/test_fr.o $(filter-out $(BUILDDIR)/./$(TESTDIR)/test_fr.o, $(OBJS))
	$(NVCC) $(DEFINES) $(CUDAFLAGS) $^ -o $@

$(BUILDDIR)/test_SpMV: $(BUILDDIR)/$(TESTDIR)/test_SpMV.o $(filter-out $(BUILDDIR)/$(TESTDIR)/test_SpMV.o, $(OBJS))
	$(NVCC) $(DEFINES) $(CUDAFLAGS) $^ -o $@

# Clean rule
clean:
	rm -rf $(BUILDDIR) $(BUILDDIR)/test_fp $(BUILDDIR)/test_fr $(BUILDDIR)/test_SpMV

# Include dependencies
-include $(OBJS:.o=.d)
