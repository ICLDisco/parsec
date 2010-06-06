CUDA_INSTALL_PATH ?= /opt/cuda-3.0
INCLUDES  := -I. -I$(CUDA_INSTALL_PATH)/include
LIB       := -L$(CUDA_INSTALL_PATH)/lib64 -lcublas -lcudart -lcuda

CXX       := g++ 
NVCC      := nvcc -cubin
#NVCC      := nvcc -cubin
LINK      := g++ -fPIC

all : sgemmN

sgemmN : sgemmN.o sgemmN-13.cubin sgemmN-11.cubin
	$(LINK) $(LIB) -o sgemmN sgemmN.o

sgemmNR : sgemmNR.o sgemmN.cubin
	$(LINK) $(LIB) -o sgemmNR sgemmNR.o

sgemm30 : sgemmN30.o sgemmN.cubin
	$(LINK) $(LIB) -o sgemm30 sgemmN30.o

sgemmN-13.cubin : sgemmN.cu
	$(NVCC) -m 32 -maxrregcount 32 -arch sm_13 sgemmN.cu -o sgemmN-13.cubin

sgemmN-11.cubin : sgemmN.cu
	$(NVCC) -m 32 -maxrregcount 32 -arch sm_11 sgemmN.cu -o sgemmN-11.cubin

sgemmN.o : sgemmN.cpp
	$(CXX) $(INCLUDES) -c sgemmN.cpp

sgemmNR.o : sgemmNR.cpp
	$(CXX) $(INCLUDES) -c sgemmNR.cpp

sgemmN30.o : sgemmN30.cpp
	$(CXX) $(INCLUDES) -c sgemmN30.cpp

clean :
	rm -f *.o *.cubin sgemmN
