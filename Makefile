GPU=0
CUDNN=0
OPENCV=0
OPENMP=1
DEBUG=0
NNPACK=1
ARCH= -gencode arch=compute_30,code=sm_30 \
      -gencode arch=compute_35,code=sm_35 \
      -gencode arch=compute_50,code=[sm_50,compute_50] \
      -gencode arch=compute_52,code=[sm_52,compute_52]
#      -gencode arch=compute_20,code=[sm_20,sm_21] \ This one is deprecated?

# This is what I use, uncomment if you know your arch and want to specify
# ARCH= -gencode arch=compute_52,code=compute_52

VPATH=./src/:./examples
SLIB=libdarknet.so
ALIB=libdarknet.a
EXEC=armdarknetNNPACKcheck1
#EXEC=armdarknetNNPACK-SVE164096128oldbuild
#EXEC=armdarknet-SVEmatrixsize
#EXEC=armdarknetNNPACK-SVEstartup
OBJDIR=./obj/

CC=~/gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-gcc
CPP=~/gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-gcc
#NVCC=nvcc 
AR=ar
ARFLAGS=rcs
#OPTS=-Ofast
LDFLAGS= -lm -pthread 
COMMON= -Iinclude/ -Isrc/ 
#CFLAGS=--target=riscv64-redhat-linux -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC -o1 -g  -mepi -v -fopenmp=libomp -fno-vectorize -fno-slp-vectorize -mllvm -no-epi-remove-redundant-vsetvl -I /root/vehave-EPI-0.7-src-seq/include/vehave-user -I /root/llvm_EPI-0.7_riscv64_native/lib
#CFLAGS=  --target=aarch64-arm-none-eabi -march=armv8-a+sve -Wall -static -O3 -Werror-implicit-function-declaration -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC 
#CFLAGS=  -march=armv8-a+sve  -S  -O0 -fno-openmp-simd -fno-signed-char -fno-simdmath -fno-vectorize -fno-slp-vectorize -disable-loop-vectorization -disable-vectorization -Wall   -Werror-implicit-function-declaration -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC -fno-vectorize -fno-slp-vectorize  

###armpl###
#CFLAGS=  -march=armv8-a+sve  -armpl=sve -mcpu=a64fx -O3  -static -fopenmp   -Wall -fsave-optimization-record -Rpass=loop-vectorize  -Werror-implicit-function-declaration -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC -L~/arm-performance-libraries_21.0_RHEL-7_gcc-10.2/arm-library/armpl_21.0_gcc-10.2/lib -I ~/arm-performance-libraries_21.0_RHEL-7_gcc-10.2/arm-library/armpl_21.0_gcc-10.2/include  #-larmpl -larmflang -lm

###onednn
CFLAGS= -march=armv8-a+sve -mcpu=generic  -O3    -static -Wall  -fsave-optimization-record   -Werror-implicit-function-declaration -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC  -I ../NNPACK/include/ -I ../NNPACK/deps/pthreadpool/include/ 
#CFLAGS=  -march=armv8-a+sve -O2   -fopenmp  -static -Wall -fsave-optimization-record -Rpass=loop-vectorize -Wl -Bdynamic  -Werror-implicit-function-declaration -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC #-larmpl -larmflang -lim
#CFLAGS=  -march=armv8-a+sve  -armpl=sve -O2 -static -fopenmp   -Wall -fsave-optimization-record -Rpass=loop-vectorize  -Werror-implicit-function-declaration -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC -L~/arm-performance-libraries_21.0_RHEL-7_gcc-10.2/arm-library/armpl_21.0_gcc-10.2/lib -I ~/arm-performance-libraries_21.0_RHEL-7_gcc-10.2/arm-library/armpl_21.0_gcc-10.2/include  #-larmpl -larmflang -lm
#CFLAGS=  -march=armv8-a+sve  -O2  -static -fopenmp -S -Wall -fsave-optimization-record -Rpass=unroll -Werror-implicit-function-declaration -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC 
#CFLAGS=  -march=armv8-a -static  -Wall -O2  -fsave-optimization-record -Rpass=unroll  -Werror-implicit-function-declaration -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC 
#CFLAGS=  -march=x86-64 -Wall -static  -O3 -fno-vectorize -fno-slp-vectorize -Werror-implicit-function-declaration -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC 
#for clang
#CFLAGS=-Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC  -v  -mllvm  
#CFLAGS=-Wall  --target=riscv64-unknown-linux-gnu -DUSE_RISCV_VECTOR -march=rv64g -static -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC -mepi -O2  -v -mllvm -no-epi-remove-redundant-vsetvl 
#CFLAGS=--target=riscv64-redhat-linux -fPIC -mepi -O1 -g -v  -fno-vectorize -fno-slp-vectorize -mllvm -no-epi-remove-redundant-vsetvl -I /root/old-vehave-EPI-src-seq/include/vehave-user -I /root/old_llvm_riscv64_native/lib

#CFLAGS=--target=riscv64-redhat-linux -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC -o1 -g -mepi -v -fopenmp=libomp -fno-vectorize -fno-slp-vectorize -mllvm -no-epi-remove-redundant-vsetvl -I /root/vehave-src-seq/include/vehave-user -I /root/llvm_riscv64_native/lib

#CFLAGS+=-fobjc-runtime=clang
ifeq ($(OPENMP), 1) 
CFLAGS+= -fopenmp
endif

ifeq ($(DEBUG), 1) 
OPTS=-O0 -g
endif

CFLAGS+=$(OPTS)
LDFLAGS+= -lstdc++
ifeq ($(OPENCV), 1) 
COMMON+= -DOPENCV
CFLAGS+= -DOPENCV
LDFLAGS+= `pkg-config --libs opencv` -lstdc++
COMMON+= `pkg-config --cflags opencv` 
endif

ifeq ($(GPU), 1) 
COMMON+= -DGPU -I/usr/local/cuda/include/
CFLAGS+= -DGPU
LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand
endif

ifeq ($(CUDNN), 1) 
COMMON+= -DCUDNN 
CFLAGS+= -DCUDNN
LDFLAGS+= -lcudnn
endif

OBJ=gemm.o utils.o cuda.o deconvolutional_layer.o convolutional_layer.o list.o image.o activations.o im2col.o col2im.o blas.o crop_layer.o dropout_layer.o maxpool_layer.o softmax_layer.o data.o matrix.o network.o connected_layer.o cost_layer.o parser.o option_list.o detection_layer.o route_layer.o upsample_layer.o box.o normalization_layer.o avgpool_layer.o layer.o local_layer.o shortcut_layer.o logistic_layer.o activation_layer.o rnn_layer.o gru_layer.o crnn_layer.o demo.o batchnorm_layer.o region_layer.o reorg_layer.o tree.o  lstm_layer.o l2norm_layer.o yolo_layer.o iseg_layer.o image_opencv.o
EXECOBJA=captcha.o lsd.o super.o art.o tag.o cifar.o go.o rnn.o segmenter.o regressor.o classifier.o coco.o yolo.o detector.o nightmare.o instance-segmenter.o darknet.o
ifeq ($(GPU), 1) 
LDFLAGS+= -lstdc++ 
OBJ+=convolutional_kernels.o deconvolutional_kernels.o activation_kernels.o im2col_kernels.o col2im_kernels.o blas_kernels.o crop_layer_kernels.o dropout_layer_kernels.o maxpool_layer_kernels.o avgpool_layer_kernels.o
endif


#LDFLAGS+= ~/NNPACK-SVE/build1/libnnpack.a ~/NNPACK-SVE/build1/deps/pthreadpool/libpthreadpool.a ~/NNPACK-SVE/build1/deps/cpuinfo/libcpuinfo_internals.a ~/NNPACK-SVE/build1/deps/cpuinfo/libcpuinfo.a
LDFLAGS+= ../NNPACK/build/libnnpack.a ../NNPACK/build/deps/pthreadpool/libpthreadpool.a ../NNPACK/build/deps/cpuinfo/libcpuinfo_internals.a ../NNPACK/build/deps/cpuinfo/libcpuinfo.a
#LDFLAGS+= /proj/snic2021-5-2/users/x_songu/build1/libnnpack.a /proj/snic2021-5-2/users/x_songu/build1/deps/pthreadpool/libpthreadpool.a /proj/snic2021-5-2/users/x_songu/build1/deps/cpuinfo/libcpuinfo_internals.a /proj/snic2021-5-2/users/x_songu/build1/deps/cpuinfo/libcpuinfo.a
#LDFLAGS+= ~/NNPACK-SVE/build1/libnnpack.a ~/NNPACK-SVE/build1/deps/pthreadpool/libpthreadpool.a /proj/snic2021-5-2/users/x_songu/build1/deps/cpuinfo/libcpuinfo_internals.a /proj/snic2021-5-2/users/x_songu/build1/deps/cpuinfo/libcpuinfo.a


EXECOBJ = $(addprefix $(OBJDIR), $(EXECOBJA))
OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile include/darknet.h

#all: obj backup results $(SLIB) $(ALIB) $(EXEC)
all: obj backup results $(ALIB) $(EXEC)
#all: obj backup results  $(ALIB) $(EXEC)
#all: obj  results $(SLIB) $(ALIB) $(EXEC)


$(EXEC): $(EXECOBJ) $(ALIB)
	$(CC)  $(CFLAGS) $(COMMON)  $^ ../NNPACK/build1/libnnpack.a -o $@  $(LDFLAGS) $(ALIB)  
#	$(CC)  $(CFLAGS) $(COMMON)  $^ ~/NNPACK-SVE/build1/libnnpack.a -o $@  $(LDFLAGS) $(ALIB)  
#	$(CC)  $(CFLAGS) $(COMMON)  $^  -o $@  $(LDFLAGS) $(ALIB)  

$(ALIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

#$(SLIB): $(OBJS)
#	$(CC) $(CFLAGS) -shared $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CPP) $(CFLAGS) $(COMMON) -c $< -o $@ 

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(CFLAGS) $(COMMON) -c $< -o $@ 

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@

obj:
	mkdir -p obj
backup:
	mkdir -p backup
results:
	mkdir -p results

.PHONY: clean

clean:
	rm -rf $(OBJS) $(SLIB) $(ALIB) $(EXEC) $(EXECOBJ) $(OBJDIR)/*

