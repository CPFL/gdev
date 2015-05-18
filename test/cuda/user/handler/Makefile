# Makefile
TARGET	= user_test
CC	= gcc
NVCC	= nvcc -arch sm_20 -cubin
LIBS	= -lenvy -lnva -lnvhw -lucuda -lgdev -lpciaccess
CFLAGS	= -std=c99 -L /usr/local/gdev/lib64 -L /usr/local/lib -I /usr/local/gdev/include -I /usr/local/include

all:
	$(NVCC) -o mmul_gpu.cubin mmul_gpu.cu
	gcc -o $(TARGET) $(CFLAGS) main.c mmul.c $(LIBS)

clean:
	rm -f $(TARGET) *.cubin ./*~
