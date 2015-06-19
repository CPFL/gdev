/*
 * Copyright (C) Shinpei Kato
 *
 * University of California, Santa Cruz
 * Systems Research Lab.
 *
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef __GDEV_NVIDIA_NVC0_H__
#define __GDEV_NVIDIA_NVC0_H__

#include "gdev_device.h"
#include "gdev_conf.h"

/* static objects. */
static struct gdev_compute gdev_compute_nvc0;

struct gdev_nvc0_query {
	uint32_t sequence;
	uint32_t pad;
	uint64_t timestamp;
};

void nvc0_compute_setup(struct gdev_device *gdev)
{
	gdev->compute = &gdev_compute_nvc0;
}

#ifdef GDEV_DEBUG
#define u64 long long unsigned int /* to avoid warnings in user-space */
static void __nvc0_launch_debug_print(struct gdev_kernel *kernel)
{
	int i;
	GDEV_PRINT("code_addr = 0x%llx\n", (u64) kernel->code_addr);
	GDEV_PRINT("code_size = 0x%llx\n", (u64) kernel->code_size);
	GDEV_PRINT("code_pc = 0x%x\n", kernel->code_pc);
	for (i = 0; i < kernel->cmem_count; i++) {
		GDEV_PRINT("cmem[%d].addr = 0x%llx\n", i, (u64) kernel->cmem[i].addr);
		GDEV_PRINT("cmem[%d].size = 0x%x\n", i, kernel->cmem[i].size);
		GDEV_PRINT("cmem[%d].offset = 0x%x\n", i, kernel->cmem[i].offset);
	}
	GDEV_PRINT("param_size = 0x%x\n", kernel->param_size);
	for (i = 0; i < kernel->param_size/4; i++)
		GDEV_PRINT("param_buf[%d] = 0x%x\n", i, kernel->param_buf[i]);
	GDEV_PRINT("lmem_addr = 0x%llx\n", (u64) kernel->lmem_addr);
	GDEV_PRINT("lmem_size_total = 0x%llx\n", (u64) kernel->lmem_size_total);
	GDEV_PRINT("lmem_size = 0x%x\n", kernel->lmem_size);
	GDEV_PRINT("lmem_size_neg = 0x%x\n", kernel->lmem_size_neg);
	GDEV_PRINT("lmem_base = 0x%x\n", kernel->lmem_base);
	GDEV_PRINT("smem_size = 0x%x\n", kernel->smem_size);
	GDEV_PRINT("smem_base = 0x%x\n", kernel->smem_base);
	GDEV_PRINT("warp_stack_size = 0x%x\n", kernel->warp_stack_size);
	GDEV_PRINT("warp_lmem_size = 0x%x\n", kernel->warp_lmem_size);
	GDEV_PRINT("reg_count = 0x%x\n", kernel->reg_count);
	GDEV_PRINT("bar_count = 0x%x\n", kernel->bar_count);
	GDEV_PRINT("grid_id = 0x%x\n", kernel->grid_id);
	GDEV_PRINT("grid_x = 0x%x\n", kernel->grid_x);
	GDEV_PRINT("grid_y = 0x%x\n", kernel->grid_y);
	GDEV_PRINT("grid_z = 0x%x\n", kernel->grid_z);
	GDEV_PRINT("block_x = 0x%x\n", kernel->block_x);
	GDEV_PRINT("block_y = 0x%x\n", kernel->block_y);
	GDEV_PRINT("block_z = 0x%x\n", kernel->block_z);
}
#endif

static int nvc0_launch(struct gdev_ctx *ctx, struct gdev_kernel *k)
{
	int x;
	uint32_t cache_split;

	/* setup cache_split so that it'll allow 3 blocks (16 warps each) per 
	   SM for maximum occupancy. */
	cache_split = k->smem_size > 16 * 1024 ? 3 : 1;

	/* local (temp) memory setup. */
	__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_COMPUTE, 0x790, 5);
	__gdev_out_ring(ctx, k->lmem_addr >> 32); /* TEMP_ADDRESS_HIGH */
	__gdev_out_ring(ctx, k->lmem_addr); /* TEMP_ADDRESS_LOW */
	__gdev_out_ring(ctx, k->lmem_size_total >> 32); /* TEMP_SIZE_HIGH */
	__gdev_out_ring(ctx, k->lmem_size_total); /* TEMP_SIZE_LOW */
	__gdev_out_ring(ctx, k->warp_lmem_size); /* WARP_TEMP_ALLOC */

	/* local memory base. */
	__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_COMPUTE, 0x77c, 1);
	__gdev_out_ring(ctx, k->lmem_base); /* LOCAL_BASE */

	/* local memory size per warp */
	__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_COMPUTE, 0x204, 3);
	__gdev_out_ring(ctx, k->lmem_size); /* LOCAL_POS_ALLOC */
	__gdev_out_ring(ctx, k->lmem_size_neg); /* LOCAL_NEG_ALLOC */
	__gdev_out_ring(ctx, k->warp_stack_size); /* WARP_CSTACK_SIZE */

	/* shared memory setup. */
	__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_COMPUTE, 0x308, 1);
	__gdev_out_ring(ctx, cache_split); /* CACHE_SPLIT */
	__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_COMPUTE, 0x214, 1);
	__gdev_out_ring(ctx, k->smem_base); /* SHARED_BASE */
	__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_COMPUTE, 0x24c, 1);
	__gdev_out_ring(ctx, k->smem_size); /* SHARED_SIZE */

	/* code flush, i.e., code needs to be uploaded in advance. */
	__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_COMPUTE, 0x1698, 1);
	__gdev_out_ring(ctx, 0x0001); /* FLUSH: 0x0001 = FLUSH_CODE */

	/* code setup. */
	__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_COMPUTE, 0x3b4, 1);
	__gdev_out_ring(ctx, (k->code_addr + k->code_pc)); /* CP_START_ID */

	/* constant memory setup. this is a bit tricky:
	   we set the constant memory size and address first. we next set
	   which const memory segment (cX[]) to be used via CB_BIND method.
	   CB_DATA will then send data (e.g., kernel parameters) to the offset
	   (CB_POS) from the constant memory address at cX[]. CB_DATA seem
	   to have 16 sockets, but not really sure how to use them... 
	   just CB_DATA#0 (0x2390) with non-increment method works here. */
	for (x = 0; x < k->cmem_count; x++) {
		if (!k->cmem[x].addr || !k->cmem[x].size)
			continue;
		__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_COMPUTE, 0x2380, 3);
		__gdev_out_ring(ctx, k->cmem[x].size); /* CB_SIZE */
		__gdev_out_ring(ctx, k->cmem[x].addr >> 32); /* CB_ADDRESS_HIGH */
		__gdev_out_ring(ctx, k->cmem[x].addr); /* CB_ADDRESS_LOW */
		__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_COMPUTE, 0x1694, 1);
		__gdev_out_ring(ctx, (x << 8) | 1); /* CB_BIND */
		/* send kernel parameters to a specific constant memory space. */
		if (x == 0) {
			int i;
			int n = k->param_size / 4; /* each param is integer size. */
			/* the following is the nvcc protocol */
			if (n >= 8) {
				k->param_buf[0] = k->smem_base;
				k->param_buf[1] = k->lmem_base;
				k->param_buf[2] = k->block_x;
				k->param_buf[3] = k->block_y;
				k->param_buf[4] = k->block_z;
				k->param_buf[5] = k->grid_x;
				k->param_buf[6] = k->grid_y;
				k->param_buf[7] = k->grid_z;
			}
			__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_COMPUTE, 0x238c, 1);
			__gdev_out_ring(ctx, k->cmem[x].offset); /* CB_POS */
			__gdev_begin_ring_nvc0_const(ctx, GDEV_SUBCH_NV_COMPUTE, 0x2390, n);
			for (i = 0; i < n; i++) {
				__gdev_out_ring(ctx, k->param_buf[i]); /* CB_DATA#0 */
			}
		}
		/* nvcc uses c1[], but what is this? */
		else if (x == 1) {
			int i;
			__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_COMPUTE, 0x238c, 1);
			__gdev_out_ring(ctx, 0); /* CB_POS */
			__gdev_begin_ring_nvc0_const(ctx, GDEV_SUBCH_NV_COMPUTE, 0x2390, 0x20);
			for (i = 0; i < 0x20; i++) {
				__gdev_out_ring(ctx, 0); /* CB_DATA#0 */
			}
			__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_COMPUTE, 0x238c, 1);
			__gdev_out_ring(ctx, 0x100); /* CB_POS */
			__gdev_begin_ring_nvc0_const(ctx, GDEV_SUBCH_NV_COMPUTE, 0x2390, 1);
			__gdev_out_ring(ctx, 0x00fffc40); /* CB_DATA#0 */
		}
	}

	/* constant memory flush. */
	__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_COMPUTE, 0x1698, 1);
	__gdev_out_ring(ctx, 0x1000); /* FLUSH: 0x1000 = FLUSH_CB */

	/* grid/block setup. */
	__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_COMPUTE, 0x238, 2);
	__gdev_out_ring(ctx, (k->grid_y << 16) | k->grid_x); /* GRIDDIM_YX */
	__gdev_out_ring(ctx, k->grid_z); /* GRIDDIM_Z */
	__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_COMPUTE, 0x3ac, 2);
	__gdev_out_ring(ctx, (k->block_y << 16) | k->block_x); /* BLOCKDIM_YX */
	__gdev_out_ring(ctx, k->block_z); /* BLOCKDIM_X */
	__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_COMPUTE, 0x250, 1);
	__gdev_out_ring(ctx, k->block_x * k->block_y * k->block_z); /* TH_ALLOC */

	/* barriers/registers setup. */
	__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_COMPUTE, 0x2c0, 1);
	__gdev_out_ring(ctx, k->reg_count); /* CP_GPR_ALLOC */
	__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_COMPUTE, 0x254, 1);
	__gdev_out_ring(ctx, k->bar_count); /* BARRIER_ALLOC */
	
	/* launch preliminary setup. */
	__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_COMPUTE, 0x780, 1);
	__gdev_out_ring(ctx, k->grid_id); /* GRIDID */
	__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_COMPUTE, 0x36c, 1);
	__gdev_out_ring(ctx, 0); /* ??? */
	__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_COMPUTE, 0x1698, 1);
	__gdev_out_ring(ctx, 0x0110); /* FLUSH: 0x110 = FLUSH_UNK8 | FLUSH_GLOBAL */
	__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_COMPUTE, 0x29c, 1);
	__gdev_out_ring(ctx, 0); /* BEGIN */
	__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_COMPUTE, 0xa08, 1);
	__gdev_out_ring(ctx, 0); /* ??? */

	/* kernel lauching. */
	__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_COMPUTE, 0x368, 1);
	__gdev_out_ring(ctx, 0x1000 /* 0x0 */); /* LAUNCH */
	__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_COMPUTE, 0xa04, 1);
	__gdev_out_ring(ctx, 0); /* END */
	__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_COMPUTE, 0x360, 1);
	__gdev_out_ring(ctx, 1); /* ??? */

	__gdev_fire_ring(ctx);

#ifdef GDEV_DEBUG
	__nvc0_launch_debug_print(k);
#endif

	return 0;
}

static uint32_t nvc0_fence_read(struct gdev_ctx *ctx, uint32_t sequence)
{
	return ((struct gdev_nvc0_query*)(ctx->fence.map))[sequence].sequence;
}

static void nvc0_fence_write(struct gdev_ctx *ctx, int subch, uint32_t sequence)
{
	uint32_t offset = sequence * sizeof(struct gdev_nvc0_query);
	uint64_t vm_addr = ctx->fence.addr + offset;
	int intr = 0; /* intr = 1 will cause an interrupt too. */

	switch (subch) {
	case GDEV_SUBCH_NV_COMPUTE:
		__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_COMPUTE, 0x110, 1);
		__gdev_out_ring(ctx, 0); /* SERIALIZE */
		__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_COMPUTE, 0x1b00, 4);
		__gdev_out_ring(ctx, vm_addr >> 32); /* QUERY_ADDRESS HIGH */
		__gdev_out_ring(ctx, vm_addr); /* QUERY_ADDRESS LOW */
		__gdev_out_ring(ctx, sequence); /* QUERY_SEQUENCE */
		__gdev_out_ring(ctx, intr << 20); /* QUERY_GET */
		break;
	case GDEV_SUBCH_NV_M2MF:
		__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_M2MF, 0x32c, 3);
		__gdev_out_ring(ctx, vm_addr >> 32); /* QUERY_ADDRESS HIGH */
		__gdev_out_ring(ctx, vm_addr); /* QUERY_ADDRESS LOW */
		__gdev_out_ring(ctx, sequence); /* QUERY_SEQUENCE */
		break;
	case GDEV_SUBCH_NV_PCOPY0:
		__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_PCOPY0, 0x338, 3);
		__gdev_out_ring(ctx, vm_addr >> 32); /* QUERY_ADDRESS HIGH */
		__gdev_out_ring(ctx, vm_addr); /* QUERY_ADDRESS LOW */
		__gdev_out_ring(ctx, sequence); /* QUERY_COUNTER */
		break;
#ifdef GDEV_NVIDIA_USE_PCOPY1
	case GDEV_SUBCH_NV_PCOPY1:
		__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_PCOPY1, 0x338, 3);
		__gdev_out_ring(ctx, vm_addr >> 32); /* QUERY_ADDRESS HIGH */
		__gdev_out_ring(ctx, vm_addr); /* QUERY_ADDRESS LOW */
		__gdev_out_ring(ctx, sequence); /* QUERY_COUNTER */
		break;
#endif
	}

	__gdev_fire_ring(ctx);
}

static void nvc0_fence_reset(struct gdev_ctx *ctx, uint32_t sequence)
{
	((struct gdev_nvc0_query*)(ctx->fence.map))[sequence].sequence = ~0;
}

static void nvc0_memcpy_m2mf(struct gdev_ctx *ctx, uint64_t dst_addr, uint64_t src_addr, uint32_t size)
{
	uint32_t mode1 = 0x102110; /* QUERY_SHORT|QUERY_YES|SRC_LINEAR|DST_LINEAR */
	uint32_t mode2 = 0x100110; /* QUERY_SHORT|SRC_LINEAR|DST_LINEAR */
	uint32_t page_size = 0x1000;
	uint32_t page_count = size / page_size;
	uint32_t rem_size = size - page_size * page_count;

	while (page_count) {
		int line_count = (page_count > 2047) ? 2047 : page_count;
		__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_M2MF, 0x238, 2);
		__gdev_out_ring(ctx, dst_addr >> 32); /* OFFSET_OUT_HIGH */
		__gdev_out_ring(ctx, dst_addr); /* OFFSET_OUT_LOW */
		__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_M2MF, 0x30c, 6);
		__gdev_out_ring(ctx, src_addr >> 32); /* OFFSET_IN_HIGH */
		__gdev_out_ring(ctx, src_addr); /* OFFSET_IN_LOW */
		__gdev_out_ring(ctx, page_size); /* SRC_PITCH_IN */
		__gdev_out_ring(ctx, page_size); /* DST_PITCH_IN */
		__gdev_out_ring(ctx, page_size); /* LINE_LENGTH_IN */
		__gdev_out_ring(ctx, line_count); /* LINE_COUNT */
		__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_M2MF, 0x300, 1);
		if (page_count == line_count && rem_size == 0)
			__gdev_out_ring(ctx, mode1); /* EXEC */
		else
			__gdev_out_ring(ctx, mode2); /* EXEC */
		page_count -= line_count;
		dst_addr += (page_size * line_count);
		src_addr += (page_size * line_count);
	}

	if (rem_size) {
		__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_M2MF, 0x238, 2);
		__gdev_out_ring(ctx, dst_addr >> 32); /* OFFSET_OUT_HIGH */
		__gdev_out_ring(ctx, dst_addr); /* OFFSET_OUT_LOW */
		__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_M2MF, 0x30c, 6);
		__gdev_out_ring(ctx, src_addr >> 32); /* OFFSET_IN_HIGH */
		__gdev_out_ring(ctx, src_addr); /* OFFSET_IN_LOW */
		__gdev_out_ring(ctx, rem_size); /* SRC_PITCH_IN */
		__gdev_out_ring(ctx, rem_size); /* DST_PITCH_IN */
		__gdev_out_ring(ctx, rem_size); /* LINE_LENGTH_IN */
		__gdev_out_ring(ctx, 1); /* LINE_COUNT */
		__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_M2MF, 0x300, 1);
		__gdev_out_ring(ctx, mode1); /* EXEC */
	}

	__gdev_fire_ring(ctx);
}

static void nvc0_memcpy_pcopy0(struct gdev_ctx *ctx, uint64_t dst_addr, uint64_t src_addr, uint32_t size)
{
	uint32_t mode = 0x3110; /* QUERY_SHORT|QUERY|SRC_LINEAR|DST_LINEAR */
	uint32_t pitch = 0x8000; /* make it configurable... */
	uint32_t ycnt = size / pitch;
	uint32_t rem_size = size - pitch * ycnt;
	size -= rem_size;

	if (size) {
		__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_PCOPY0, 0x30c, 6);
		__gdev_out_ring(ctx, src_addr >> 32); /* SRC_ADDRESS_HIGH */
		__gdev_out_ring(ctx, src_addr); /* SRC_ADDRESS_LOW */
		__gdev_out_ring(ctx, dst_addr >> 32); /* DST_ADDRESS_HIGH */
		__gdev_out_ring(ctx, dst_addr); /* DST_ADDRESS_LOW */
		__gdev_out_ring(ctx, pitch); /* SRC_PITCH */
		__gdev_out_ring(ctx, pitch); /* DST_PITCH */
		__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_PCOPY0, 0x324, 2);
		__gdev_out_ring(ctx, pitch); /* XCNT */
		__gdev_out_ring(ctx, ycnt); /* YCNT */
		__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_PCOPY0, 0x300, 1);
		__gdev_out_ring(ctx, mode); /* EXEC */

		__gdev_fire_ring(ctx);
	}

	if (rem_size) {
		src_addr += size;
		dst_addr += size;
		__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_PCOPY0, 0x30c, 6);
		__gdev_out_ring(ctx, src_addr >> 32); /* SRC_ADDRESS_HIGH */
		__gdev_out_ring(ctx, src_addr); /* SRC_ADDRESS_LOW */
		__gdev_out_ring(ctx, dst_addr >> 32); /* DST_ADDRESS_HIGH */
		__gdev_out_ring(ctx, dst_addr); /* DST_ADDRESS_LOW */
		__gdev_out_ring(ctx, 0); /* SRC_PITCH */
		__gdev_out_ring(ctx, 0); /* DST_PITCH */
		__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_PCOPY0, 0x324, 2);
		__gdev_out_ring(ctx, rem_size); /* XCNT */
		__gdev_out_ring(ctx, 1); /* YCNT */
		__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_PCOPY0, 0x300, 1);
		__gdev_out_ring(ctx, mode); /* EXEC */

		__gdev_fire_ring(ctx);
	}
}

static void nvc0_membar(struct gdev_ctx *ctx)
{
	/* this must be a constant method. */
	__gdev_begin_ring_nvc0_const(ctx, GDEV_SUBCH_NV_COMPUTE, 0x21c, 2);
	__gdev_out_ring(ctx, 4); /* MEM_BARRIER */
	__gdev_out_ring(ctx, 0x1111); /* maybe wait for everything? */

	__gdev_fire_ring(ctx);
}

static void nvc0_notify_intr(struct gdev_ctx *ctx)
{
	uint64_t addr = ctx->notify.addr;

	__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_COMPUTE, 0x110, 1);
	__gdev_out_ring(ctx, 0); /* SERIALIZE */
	__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_COMPUTE, 0x104, 3);
	__gdev_out_ring(ctx, addr >> 32); /* NOTIFY_HIGH_ADDRESS */
	__gdev_out_ring(ctx, addr); /* NOTIFY_LOW_ADDRESS */
	__gdev_out_ring(ctx, 1); /* WRITTEN_AND_AWAKEN */
	__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_COMPUTE, 0x100, 1);
	__gdev_out_ring(ctx, ctx->cid); /* NOP */

	__gdev_fire_ring(ctx);
}

static void nvc0_init(struct gdev_ctx *ctx)
{
	int i;
	uint64_t mp_limit;
	struct gdev_vas *vas = ctx->vas;
	struct gdev_device *gdev = vas->gdev;

	/* initialize the fence values. */
	for (i = 0; i < GDEV_FENCE_COUNT; i++)
		nvc0_fence_reset(ctx, i);

	/* clean the FIFO. */
	for (i = 0; i < 128/4; i++)
		__gdev_out_ring(ctx, 0);
	__gdev_fire_ring(ctx);

	/* setup subchannels. */
	__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_M2MF, 0, 1);
	__gdev_out_ring(ctx, 0x9039); /* M2MF */
	__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_COMPUTE, 0, 1);
	__gdev_out_ring(ctx, 0x90c0); /* COMPUTE */

	/* enable PCOPY only when we are in the kernel atm... */
#ifdef __KERNEL__
#if LINUX_VERSION_CODE < KERNEL_VERSION(3,7,0)
	__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_PCOPY0, 0, 1);
	__gdev_out_ring(ctx, 0x490b5); /* PCOPY0 */
#ifdef GDEV_NVIDIA_USE_PCOPY1
	__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_PCOPY1, 0, 1);
	__gdev_out_ring(ctx, 0x590b8); /* PCOPY1 */
#endif
#endif
#endif
	__gdev_fire_ring(ctx);

	/* the blob places NOP at the beginning. */
	__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_COMPUTE, 0x100, 1);
	__gdev_out_ring(ctx, 0); /* GRAPH_NOP */

	/* hardware limit. */
	gdev_query(gdev, GDEV_NVIDIA_QUERY_MP_COUNT, &mp_limit);
	__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_COMPUTE, 0x758, 1);
	__gdev_out_ring(ctx, (uint32_t) mp_limit); /* MP_LIMIT */
	__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_COMPUTE, 0xd64, 1);
	__gdev_out_ring(ctx, 0xf); /* CALL_LIMIT_LOG: hardcoded for now */

	/* grid/block initialization. the blob does the following, but not 
	   really sure if they are necessary... */
	__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_COMPUTE, 0x2a0, 1);
	__gdev_out_ring(ctx, 0x8000); /* ??? */
	__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_COMPUTE, 0x238, 2);
	__gdev_out_ring(ctx, (1 << 16) | 1); /* GRIDDIM_YX */
	__gdev_out_ring(ctx, 1); /* GRIDDIM_Z */
	__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_COMPUTE, 0x3ac, 2);
	__gdev_out_ring(ctx, (1 << 16) | 1); /* BLOCKDIM_YX */
	__gdev_out_ring(ctx, 1); /* BLOCKDIM_X */

	/* global memory setup: 0xc << 28 = read_ok & write_ok. 
	   HIGH_MASK = 0x000000ff (x << 0) and INDEX_MASK = 0x00ff0000 (x << 16).
	   this will remap high bytes of g[], to the actual global memory address.
	   e.g., if INDEX = 0xff and HIGH = 0x00, g[0xff000004] in the kernel
	   program will reference address 0x4. */
	__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_COMPUTE, 0x2c4, 1);
	__gdev_out_ring(ctx, 0); /* ???: UNK2C4 <- FALSE */
	for (i = 0; i < 0xff; i++) {
		__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_COMPUTE, 0x2c8, 1);
		__gdev_out_ring(ctx, (0xc << 28) | (i << 16) | i); /* GLOBAL_BASE */
	}
	__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_COMPUTE, 0x2c4, 1);
	__gdev_out_ring(ctx, 1); /* ???: UNK2C4 <- TRUE */

#ifdef GDEV_TEXTURE_SUPPORT /* not supported now... */
	/* texture setup. hardcode samp_log2 = tex_log2 = 3... FIXME!!! */
	__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_COMPUTE, 0x210, 1);
	__gdev_out_ring(ctx, 0x33); /* TEX_LIMITS */
	__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_COMPUTE, 0x1234, 1);
	__gdev_out_ring(ctx, 1); /* LINKED_TSC */
	__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_COMPUTE, 0x1578, 3);
	__gdev_out_ring(ctx, 0); /* TIC_ADDRESS_HIGH */
	__gdev_out_ring(ctx, 0); /* TIC_ADDRESS_LOW */
	__gdev_out_ring(ctx, 0x3ff); /* TIC_LIMIT */
	__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_COMPUTE, 0x155c, 3);
	__gdev_out_ring(ctx, 0); /* TSC_ADDRESS_HIGH */
	__gdev_out_ring(ctx, 0); /* TSC_ADDRESS_LOW */
	__gdev_out_ring(ctx, 0x3ff); /* TSC_LIMIT */
#endif

	__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_COMPUTE, 0x1608, 2);
	__gdev_out_ring(ctx, 0x0); /* CODE_ADDRESS_HIGH */
	__gdev_out_ring(ctx, 0x0); /* CODE_ADDRESS_LOW */

	__gdev_begin_ring_nvc0(ctx, GDEV_SUBCH_NV_COMPUTE, 0x1698, 1);
	__gdev_out_ring(ctx, 0x0001); /* FLUSH: 0x0001 = FLUSH_CODE */

	__gdev_fire_ring(ctx);
}

static struct gdev_compute gdev_compute_nvc0 = {
	.launch = nvc0_launch,
	.fence_read = nvc0_fence_read,
	.fence_write = nvc0_fence_write,
	.fence_reset = nvc0_fence_reset,
	.memcpy = nvc0_memcpy_m2mf,
	.memcpy_async = nvc0_memcpy_pcopy0,
	.membar = nvc0_membar,
	.notify_intr = nvc0_notify_intr,
	.init = nvc0_init,
};

#endif
