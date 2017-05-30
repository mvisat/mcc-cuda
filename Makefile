TARGET    := mcc_cuda
SRC_DIR   := src
OBJ_DIR   := obj
BUILD_DIR := build

CUDA_PATH := /usr/local/cuda
LDFLAGS	  += -lcuda -lcudart -lm -lrt
INCLUDES  += -I. -I$(CUDA_PATH)/include
LIBS      += -L$(CUDA_PATH)/lib64
CXXFLAGS  += -Wall -pedantic -O3 -std=c++11
NVCC			:= nvcc
NVCCFLAGS := -O3 -std=c++11 --ptxas-options=-v
CUDA_ARCH := -arch=sm_61

CPP_FILES := $(wildcard $(SRC_DIR)/*.cpp)
CU_FILES  := $(wildcard $(SRC_DIR)/*.cu)

H_FILES   := $(wildcard $(SRC_DIR)/*.h)
CUH_FILES := $(wildcard $(SRC_DIR)/*.cuh)

OBJ_FILES := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_FILES))
CUO_FILES := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.cu.o,$(CU_FILES))

$(TARGET): $(OBJ_FILES) $(CUO_FILES)
	$(CXX) -o $(BUILD_DIR)/$@ $? $(INCLUDES) $(LIBS) $(LDFLAGS)

$(OBJ_DIR)/%.cu.o: $(SRC_DIR)/%.cu $(CUH_FILES)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -c -o $@ $<

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp $(H_FILES)
	$(CXX) $(CXXFLAGS) -c -o $@ $< $(INCLUDES)

.PHONY: clean
clean:
	rm -f $(OBJ_DIR)/*
	rm -f $(BUILD_DIR)/*

run: $(TARGET)
	$(BUILD_DIR)/$(TARGET)
