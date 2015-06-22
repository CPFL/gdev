#include <stdint.h>
#include <cuda.h>
__global__
void add(uint32_t *a, uint32_t *b, uint32_t *c, uint32_t n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n) {
	    for (int k = 0; k < (8192 * 256); ++k) {
		int idx = i * n + j;
		c[idx] = a[idx] + b[idx];
	    }
    }
}
