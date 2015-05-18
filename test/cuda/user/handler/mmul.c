#include <cuda.h>
#ifdef __KERNEL__ /* just for measurement */
#include <linux/vmalloc.h>
#include <linux/time.h>
#define printf printk
#define malloc vmalloc
#define free vfree
#define gettimeofday(x, y) do_gettimeofday(x)
#else /* just for measurement */
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#endif

#include <stdint.h>
#include <assert.h>
#include <pciaccess.h>
#include <unistd.h>
#include <envytools/nva.h>
#include <envytools/util.h>

void dump_bpt(void)
{
	unsigned gpcs  = nva_rd32(0, 0x409604) & 0xff;
	for (unsigned i = 0; i < gpcs; ++i) {
		unsigned gpc_base = 0x500000 + 0x8000 * i;
		unsigned tpcs = nva_rd32(0, 0x502608 + i * 0x8000) & 0xff;
		for (unsigned j = 0; j < tpcs; ++j) {
			unsigned tpc_base = gpc_base + 0x4000 + 0x800 * j;
			unsigned mp_base = tpc_base + 0x600;
			unsigned bpt = nva_rd32(0, mp_base + 0x10);
			printf("GPC:(%u),TPC:(%u),BASE:(%x),MP:(%x),BPT:(%x)\n", i, j, tpc_base, mp_base, bpt);
		}
	}
}

void trigger(void)
{
	// Trigger trap in broadcast area.
	unsigned gpc_base = 0x418000;
	unsigned tpc_base = gpc_base + 0x1800;
	unsigned mp_base = tpc_base + 0x600;
	unsigned bpt_control = mp_base + 0x10;
	// unsigned bpt = 0x80000001;
	unsigned bpt = 0x80000001;
	printf("TRIGGER BPT:(%x)\n", nva_rd32(0, mp_base + 0x10));
	// Fire trigger. (1U << 31) | 1U.
	nva_wr32(0, mp_base + 0x10, bpt);

	unsigned gpcs  = nva_rd32(0, 0x409604) & 0xff;
	printf("GPCS:(%u)\n", gpcs);  // 4
	for (unsigned i = 0; i < gpcs; ++i) {
		unsigned gpc_base = 0x500000 + 0x8000 * i;
		unsigned tpcs = nva_rd32(0, 0x502608 + i * 0x8000) & 0xff;
		printf("GPC:(%u),TPCS:(%u)\n", i, tpcs);
		for (unsigned j = 0; j < tpcs; ++j) {
			unsigned tpc_base = gpc_base + 0x4000 + 0x800 * j;
			unsigned mp_base = tpc_base + 0x600;
			unsigned bpt = nva_rd32(0, mp_base + 0x10);
			printf("GPC:(%u),TPC:(%u),BASE:(%x),MP:(%x)\n", i, j, tpc_base, mp_base);
			printf("BPT:(%x)\n", bpt);
			nva_wr32(0, mp_base + 0x10, bpt);
			printf("BPT:(%x)\n", nva_rd32(0, mp_base + 0x10));
		}
	}
}

void enable_debugging(void)
{
	unsigned gpcs  = nva_rd32(0, 0x409604) & 0xff;
	printf("GPCS:(%u)\n", gpcs);  // 4
	for (unsigned i = 0; i < gpcs; ++i) {
		unsigned gpc_base = 0x500000 + 0x8000 * i;
		unsigned tpcs = nva_rd32(0, 0x502608 + i * 0x8000) & 0xff;
		printf("GPC:(%u),TPCS:(%u)\n", i, tpcs);
		for (unsigned j = 0; j < tpcs; ++j) {
			unsigned tpc_base = gpc_base + 0x4000 + 0x800 * j;
			unsigned mp_base = tpc_base + 0x600;
			printf("GPC:(%u),TPC:(%u),BASE:(%x),MP:(%x)\n", i, j, tpc_base, mp_base);
			printf("BPT:(%x)\n", nva_rd32(0, mp_base + 0x10));
			nva_wr32(0, mp_base + 0x10, 0x1);
			printf("BPT:(%x)\n", nva_rd32(0, mp_base + 0x10));
		}
	}
}

int pci_main(void)
{
	int ret = nva_init();
	assert(ret == 0);
	assert(nva_cardsnum == 1);
	// struct nva_card* card = nva_cards[0];
	unsigned gpcs  = nva_rd32(0, 0x409604) & 0xff;
	printf("GPCS:(%u)\n", gpcs);  // 4
	for (unsigned i = 0; i < gpcs; ++i) {
		unsigned gpc_base = 0x500000 + 0x8000 * i;
		unsigned tpcs = nva_rd32(0, 0x502608 + i * 0x8000) & 0xff;
		printf("GPC:(%u),TPCS:(%u)\n", i, tpcs);
		for (unsigned j = 0; j < tpcs; ++j) {
			unsigned tpc_base = gpc_base + 0x4000 + 0x800 * j;
			unsigned mp_base = tpc_base + 0x600;
			printf("GPC:(%u),TPC:(%u),BASE:(%x),MP:(%x)\n", i, j, tpc_base, mp_base);
			printf("BPT:(%x)\n", nva_rd32(0, mp_base + 0x10));
			printf("BPT:(%x)\n", nva_rd32(0, mp_base + 0x10));
		}
	}

	// Enable debugging mode.
	// enable_debugging();
}

/* tvsub: ret = x - y. */
static inline void tvsub(struct timeval *x, 
						 struct timeval *y, 
						 struct timeval *ret)
{
	ret->tv_sec = x->tv_sec - y->tv_sec;
	ret->tv_usec = x->tv_usec - y->tv_usec;
	if (ret->tv_usec < 0) {
		ret->tv_sec--;
		ret->tv_usec += 1000000;
	}
}

int cuda_test_mmul(unsigned int n, char *path)
{
	int i, j, idx;
	CUresult res;
	CUdevice dev;
	CUcontext ctx;
	CUfunction function;
	CUfunction handler;
	CUmodule module;
	CUdeviceptr a_dev, b_dev, c_dev;
	CUdeviceptr handler_dev;
	unsigned int *a = (unsigned int *) malloc (n*n * sizeof(unsigned int));
	unsigned int *b = (unsigned int *) malloc (n*n * sizeof(unsigned int));
	unsigned int *c = (unsigned int *) malloc (n*n * sizeof(unsigned int));
	int block_x, block_y, grid_x, grid_y;
	char fname[256];
	struct timeval tv;
	struct timeval tv_total_start, tv_total_end;
	float total;
	struct timeval tv_h2d_start, tv_h2d_end;
	float h2d;
	struct timeval tv_d2h_start, tv_d2h_end;
	float d2h;
	struct timeval tv_exec_start, tv_exec_end;
	struct timeval tv_mem_alloc_start;
	struct timeval tv_data_init_start;
	float data_init;
	struct timeval tv_conf_kern_start;
	struct timeval tv_close_start;
	float mem_alloc;
	float exec;
	float init_gpu;
	float configure_kernel;
	float close_gpu;
	float data_read;

	unsigned int dummy_b, dummy_c;

	pci_main();

	/* block_x * block_y should not exceed 512. */
	block_x = n < 16 ? n : 16;
	block_y = n < 16 ? n : 16;
	grid_x = n / block_x;
	if (n % block_x != 0)
		grid_x++;
	grid_y = n / block_y;
	if (n % block_y != 0)
		grid_y++;
	printf("block = (%d, %d)\n", block_x, block_y);
	printf("grid = (%d, %d)\n", grid_x, grid_y);

	gettimeofday(&tv_total_start, NULL);

	res = cuInit(0);
	if (res != CUDA_SUCCESS) {
		printf("cuInit failed: res = %lu\n", (unsigned long)res);
		return -1;
	}

	res = cuDeviceGet(&dev, 0);
	if (res != CUDA_SUCCESS) {
		printf("cuDeviceGet failed: res = %lu\n", (unsigned long)res);
		return -1;
	}

	dump_bpt();
	res = cuCtxCreate(&ctx, 0, dev);
	dump_bpt();
	// enable_debugging();
	dump_bpt();
	if (res != CUDA_SUCCESS) {
		printf("cuCtxCreate failed: res = %lu\n", (unsigned long)res);
		return -1;
	}

	sprintf(fname, "%s/mmul_gpu.cubin", path);
	res = cuModuleLoad(&module, fname);
	if (res != CUDA_SUCCESS) {
		printf("cuModuleLoad() failed\n");
		return -1;
	}
	res = cuModuleGetFunction(&function, module, "multiply");
	if (res != CUDA_SUCCESS) {
		printf("cuModuleGetFunction() failed\n");
		return -1;
	}
// 	res = cuModuleGetFunction(&handler, module, "_Z7handlerv");
// 	if (res != CUDA_SUCCESS) {
// 		printf("cuModuleGetFunction() failed\n");
// 		return -1;
// 	}
//
// 	res = cuFuncGetCodeAddr(&handler_dev, handler);
// 	if (res != CUDA_SUCCESS) {
// 		printf("cuFuncGetCodeAddr() failed\n");
// 		return -1;
// 	}
//
// 	unsigned long long handler_content[128] = { 0 };
//
// 	res = cuMemcpyDtoH(handler_content, handler_dev, 128 * sizeof(unsigned long long));
// 	if (res != CUDA_SUCCESS) {
// 		printf("cuMemcpyHtoD (a) failed: res = %lu\n", (unsigned long)res);
// 		return -1;
// 	}
//
// 	printf("HANDLER\n");
// 	for (int i = 0; i < 128; ++i) {
// 		printf("    0x%llx\n", handler_content[i]);
// 	}
// 	printf("HANDLER END\n");


/*	res = cuFuncSetSharedSize(function, 0x40); /* just random 
	if (res != CUDA_SUCCESS) {
		printf("cuFuncSetSharedSize() failed\n");
		return -1;
	}
*/
	res = cuFuncSetBlockShape(function, block_x, block_y, 1);
	if (res != CUDA_SUCCESS) {
		printf("cuFuncSetBlockShape() failed\n");
		return -1;
	}

	gettimeofday(&tv_mem_alloc_start, NULL);


	/* a[] */
	res = cuMemAlloc(&a_dev, n*n * sizeof(unsigned int));
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc (a) failed\n");
		return -1;
	}
	/* b[] */
	res = cuMemAlloc(&b_dev, n*n * sizeof(unsigned int));
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc (b) failed\n");
		return -1;
	}
	/* c[] */
	res = cuMemAlloc(&c_dev, n*n * sizeof(unsigned int));
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc (c) failed\n");
		return -1;
	}

	gettimeofday(&tv_data_init_start, NULL);

	/* initialize A[] & B[] */
	for (i = 0; i < n; i++) {
		idx = i*n;
		for(j = 0; j < n; j++) {			
			a[idx++] = i;
		}
	}
	for (i = 0; i < n; i++) {
		idx = i*n;
		for(j = 0; j < n; j++) {
			b[idx++] = i;
		}
	}


	gettimeofday(&tv_h2d_start, NULL);
	/* upload a[] and b[] */
	res = cuMemcpyHtoD(a_dev, a, n*n * sizeof(unsigned int));
	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyHtoD (a) failed: res = %lu\n", (unsigned long)res);
		return -1;
	}
	res = cuMemcpyHtoD(b_dev, b, n*n * sizeof(unsigned int));
	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyHtoD (b) failed: res = %lu\n", (unsigned long)res);
		return -1;
	}
	gettimeofday(&tv_h2d_end, NULL);

	gettimeofday(&tv_conf_kern_start, NULL);

	/* set kernel parameters */
	res = cuParamSeti(function, 0, a_dev);	
	if (res != CUDA_SUCCESS) {
		printf("cuParamSeti (a) failed: res = %lu\n", (unsigned long)res);
		return -1;
	}
	res = cuParamSeti(function, 4, a_dev >> 32);
	if (res != CUDA_SUCCESS) {
		printf("cuParamSeti (a) failed: res = %lu\n", (unsigned long)res);
		return -1;
	}
	res = cuParamSeti(function, 8, b_dev);
	if (res != CUDA_SUCCESS) {
		printf("cuParamSeti (b) failed: res = %lu\n", (unsigned long)res);
		return -1;
	}
	res = cuParamSeti(function, 12, b_dev >> 32);
	if (res != CUDA_SUCCESS) {
		printf("cuParamSeti (b) failed: res = %lu\n", (unsigned long)res);
		return -1;
	}
	res = cuParamSeti(function, 16, c_dev);
	if (res != CUDA_SUCCESS) {
		printf("cuParamSeti (c) failed: res = %lu\n", (unsigned long)res);
		return -1;
	}
	res = cuParamSeti(function, 20, c_dev >> 32);
	if (res != CUDA_SUCCESS) {
		printf("cuParamSeti (c) failed: res = %lu\n", (unsigned long)res);
		return -1;
	}
	res = cuParamSeti(function, 24, n);
	if (res != CUDA_SUCCESS) {
		printf("cuParamSeti (c) failed: res = %lu\n", (unsigned long)res);
		return -1;
	}
	res = cuParamSetSize(function, 28);
	if (res != CUDA_SUCCESS) {
		printf("cuParamSetSize failed: res = %lu\n", (unsigned long)res);
		return -1;
	}
// 	res = cuParamSetHandler(function, handler);
// 	if (res != CUDA_SUCCESS) {
// 		printf("cuParamSetSize failed: res = %lu\n", (unsigned long)res);
// 		return -1;
// 	}

	gettimeofday(&tv_exec_start, NULL);
	/* launch the kernel */
	dump_bpt();
	res = cuLaunchGrid(function, grid_x, grid_y);
	if (res != CUDA_SUCCESS) {
		printf("cuLaunchGrid failed: res = %lu\n", (unsigned long)res);
		return -1;
	}
	unsigned gpc_base = 0x418000;
	unsigned tpc_base = gpc_base + 0x1800;
	unsigned mp_base = tpc_base + 0x600;
	unsigned bpt_control = mp_base + 0x10;
	unsigned bpt = 0x80000001;
	unsigned bpt_unpause = 0x40000001;
	printf("status %lx\n", nva_rd32(0, mp_base + 0x0c));


	printf("prev trigger\n");
	trigger();
	printf("trigger!\n");


	printf("status %lx\n", nva_rd32(0, mp_base + 0x0c));
	printf("unpause\n");
	// nva_wr32(0, bpt_control, bpt_unpause);
	printf("unpaused!\n");
	cuCtxSynchronize();
	// dump_bpt();
	gettimeofday(&tv_exec_end, NULL);

	gettimeofday(&tv_d2h_start, NULL);
	/* download c[] */
	res = cuMemcpyDtoH(c, c_dev, n*n * sizeof(unsigned int));
	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyDtoH (c) failed: res = %lu\n", (unsigned long)res);
		return -1;
	}
	gettimeofday(&tv_d2h_end, NULL);

	gettimeofday(&tv_close_start, NULL);

	res = cuMemFree(a_dev);
	if (res != CUDA_SUCCESS) {
		printf("cuMemFree (a) failed: res = %lu\n", (unsigned long)res);
		return -1;
	}
	res = cuMemFree(b_dev);
	if (res != CUDA_SUCCESS) {
		printf("cuMemFree (b) failed: res = %lu\n", (unsigned long)res);
		return -1;
	}
	res = cuMemFree(c_dev);
	if (res != CUDA_SUCCESS) {
		printf("cuMemFree (c) failed: res = %lu\n", (unsigned long)res);
		return -1;
	}

	res = cuModuleUnload(module);
	if (res != CUDA_SUCCESS) {
		printf("cuModuleUnload failed: res = %lu\n", (unsigned long)res);
		return -1;
	}

	res = cuCtxDestroy(ctx);
	if (res != CUDA_SUCCESS) {
		printf("cuCtxDestroy failed: res = %lu\n", (unsigned long)res);
		return -1;
	}

	free(a);
	free(b);
	free(c);

	gettimeofday(&tv_total_end, NULL);


	tvsub(&tv_mem_alloc_start, &tv_total_start, &tv);
	init_gpu = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

	tvsub(&tv_data_init_start, &tv_mem_alloc_start, &tv);
	mem_alloc = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

	tvsub(&tv_h2d_start, &tv_data_init_start, &tv);
	data_init = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

	tvsub(&tv_h2d_end, &tv_h2d_start, &tv);
	h2d = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

	tvsub(&tv_exec_start, &tv_conf_kern_start, &tv);
	configure_kernel = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

	tvsub(&tv_exec_end, &tv_exec_start, &tv);
	exec = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

	tvsub(&tv_d2h_end, &tv_d2h_start, &tv);
	d2h = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

	tvsub(&tv_close_start, &tv_d2h_end, &tv);
	data_read = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

	tvsub(&tv_total_end, &tv_close_start, &tv);
	close_gpu = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

	tvsub(&tv_total_end, &tv_total_start, &tv);
	total = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

	printf("Init: %f\n", init_gpu);
	printf("MemAlloc: %f\n", mem_alloc);
	printf("DataInit: %f\n", data_init);
	printf("HtoD: %f\n", h2d);
	printf("KernConf: %f\n", configure_kernel);
	printf("Exec: %f\n", exec);
	printf("DtoH: %f\n", d2h);
	printf("DataRead: %f\n", data_read);
	printf("Close: %f\n", close_gpu);
	printf("Total: %f\n", total);

	return 0;
}


