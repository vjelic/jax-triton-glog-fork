# -*- coding: utf-8 -*-


# Imports.
# ------------------------------------------------------------------------------

# Python standard library
import typing

# Triton
import triton
import triton.language as tl


# Triton GMM kernel.
# ------------------------------------------------------------------------------


@triton.jit
@typing.no_type_check
def triton_gmm_kernel_core(
    # Tensor pointers:
    lhs_ptr,
    rhs_ptr,
    group_sizes_ptr,
    out_ptr,
    # Tensor shapes:
    M: int,
    K: int,
    N: int,
    G: int,
    # Tensor strides:
    stride_lhs_m: int,
    stride_lhs_k: int,
    stride_rhs_g: int,
    stride_rhs_k: int,
    stride_rhs_n: int,
    stride_out_m: int,
    stride_out_n: int,
    # Meta-parameters:
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    K_DIVISIBLE_BY_BLOCK_SIZE_K: tl.constexpr,
):
    #tl.assume(M > 0)
    #tl.assume(K > 0)
    #tl.assume(N > 0)
    #tl.assume(G > 0)

    #tl.assume(stride_lhs_m > 0)
    #tl.assume(stride_lhs_k > 0)
    #tl.assume(stride_rhs_g > 0)
    #tl.assume(stride_rhs_k > 0)
    #tl.assume(stride_rhs_n > 0)
    #tl.assume(stride_out_m > 0)
    #tl.assume(stride_out_n > 0)

    stride_lhs_type = tl.int64
    stride_rhs_type = tl.int64
    stride_out_type = tl.int64

    '''
    stride_lhs_m = stride_lhs_m.to(stride_lhs_type)
    stride_lhs_k = stride_lhs_k.to(stride_lhs_type)
    stride_rhs_g = stride_rhs_g.to(stride_rhs_type)
    stride_rhs_k = stride_rhs_k.to(stride_rhs_type)
    stride_rhs_n = stride_rhs_n.to(stride_rhs_type)
    stride_out_m = stride_out_m.to(stride_out_type)
    stride_out_n = stride_out_n.to(stride_out_type)
    '''

    # Current tile. Each program computes multiple tiles of each group.
    tile = tl.program_id(0)
    #tl.assume(tile >= 0)

    # Tile limit of last MM problem (inclusive).
    last_mm_tile = 0

    # Last input row of lhs and output row of out. Each group reads some rows of
    # lhs and writes some rows to out.
    last_row = 0

    # Loop through all (m, K, N) MM problems:
    #   (m, K) x (K, N) = (m, N)
    #   sum(m) = M
    for g in range(G):
        # Get m dimension of current MM problem.
        m = tl.load(group_sizes_ptr + g)
        # m can be zero if group is empty
        #tl.assume(m >= 0)

        num_m_tiles = tl.cdiv(m, BLOCK_SIZE_M)
        # num_m_tiles can be zero if group is empty
        #tl.assume(num_m_tiles >= 0)
        num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
        #tl.assume(num_n_tiles > 0)
        num_tiles = num_m_tiles * num_n_tiles
        # num_tiles can be zero if group is empty
        #tl.assume(num_tiles >= 0)

        # Loop through tiles of current MM problem.
        while tile >= last_mm_tile and tile < last_mm_tile + num_tiles:
            # Figure out tile coordinates in current MM problem.
            tile_in_mm = tile - last_mm_tile
            #tl.assume(tile_in_mm >= 0)
            tile_m = tile_in_mm // num_n_tiles
            #tl.assume(tile_m >= 0)
            #tl.assume(tile_m < num_m_tiles)
            tile_n = tile_in_mm % num_n_tiles
            #tl.assume(tile_n >= 0)
            #tl.assume(tile_n < num_n_tiles)

            # Do regular MM:

            #tl.assume(tile_m * BLOCK_SIZE_M >= 0)
            #tl.assume(tile_n * BLOCK_SIZE_N >= 0)

            offs_lhs_m = (
                tile_m.to(stride_lhs_type) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            ) % m
            #tl.assume( offs_lhs_m.dtype == stride_lhs_type)
            offs_rhs_n = (
                tile_n.to(stride_rhs_type) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            ) % N
            #tl.assume( offs_rhs_n.dtype == stride_lhs_type )
            offs_k = tl.arange(0, BLOCK_SIZE_K)

            lhs_offs_0 = last_row + offs_lhs_m[:, None]
            #tl.assume( lhs_offs_0.dtype == stride_lhs_type)
            lhs_offs_1 = lhs_offs_0 * stride_lhs_m
            #tl.assume(lhs_offs_1.dtype == stride_lhs_type)
            lhs_offs_2 = offs_k[None, :] * stride_lhs_k
            #tl.assume(lhs_offs_2.dtype == stride_lhs_type )
            lhs_offs_3 = lhs_offs_1 + lhs_offs_2
            #tl.assume(lhs_offs_3.dtype == stride_lhs_type)
            lhs_ptrs = lhs_ptr + lhs_offs_3

            rhs_offs_1 = g * stride_rhs_g
            #tl.assume(rhs_offs_1.dtype == stride_rhs_type)
            rhs_offs_2 = offs_k[:, None] * stride_rhs_k
            #tl.assume(rhs_offs_2.dtype == stride_rhs_type)
            rhs_offs_3 = offs_rhs_n[None, :] * stride_rhs_n
            #tl.assume(rhs_offs_3.dtype == stride_rhs_type)
            rhs_offs_4 = rhs_offs_1 + rhs_offs_2 + rhs_offs_3
            #tl.assume(rhs_offs_4.dtype == stride_rhs_type)
            rhs_ptrs = rhs_ptr + rhs_offs_4

            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                if K_DIVISIBLE_BY_BLOCK_SIZE_K:
                    lhs = tl.load(lhs_ptrs)
                    rhs = tl.load(rhs_ptrs)
                else:
                    k_mask_limit = K - k * BLOCK_SIZE_K
                    lhs = tl.load(
                        lhs_ptrs, mask=offs_k[None, :] < k_mask_limit, other=0
                    )
                    rhs = tl.load(
                        rhs_ptrs, mask=offs_k[:, None] < k_mask_limit, other=0
                    )

                acc += tl.dot(lhs, rhs, input_precision="ieee")

                lhs_step = BLOCK_SIZE_K * stride_lhs_k
                #tl.assume(lhs_step > 0)
                #tl.assume( lhs_step.dtype == stride_lhs_type)
                lhs_ptrs += lhs_step

                rhs_step = BLOCK_SIZE_K * stride_rhs_k
                #tl.assume(rhs_step > 0)
                #tl.assume(rhs_step.dtype == stride_rhs_type)
                rhs_ptrs += rhs_step

            acc = acc.to(out_ptr.type.element_ty)

            offs_out_m = tile_m.to(stride_out_type) * BLOCK_SIZE_M + tl.arange(
                0, BLOCK_SIZE_M
            )
            #tl.assume(offs_out_m.dtype == stride_out_type)
            offs_out_n = tile_n.to(stride_out_type) * BLOCK_SIZE_N + tl.arange(
                0, BLOCK_SIZE_N
            )
            #tl.assume(offs_out_n.dtype == stride_out_type)

            out_offs_0 = last_row + offs_out_m[:, None]
            #tl.assume( out_offs_0.dtype == stride_out_type)
            out_offs_1 = out_offs_0 * stride_out_m
            #tl.assume( out_offs_1.dtype == stride_out_type)
            out_offs_2 = offs_out_n[None, :] * stride_out_n
            #tl.assume( out_offs_2.dtype == stride_out_type)
            out_offs_3 = out_offs_1 + out_offs_2
            #tl.assume(out_offs_3.dtype == stride_out_type)
            out_ptrs = out_ptr + out_offs_3

            tl.store(
                out_ptrs,
                acc,
                mask=(offs_out_m[:, None] < m) & (offs_out_n[None, :] < N),
            )

            # Go to the next tile by advancing number of programs.
            tile += tl.num_programs(0)
            #tl.assume(tile > 0)

        # Get ready to go to the next MM problem.
        last_mm_tile += num_tiles
        # last_mm_tile can be zero if group 0 is skipped
        #tl.assume(last_mm_tile >= 0)
        last_row += m
        # last_row can be zero if group 0 is skipped
        #tl.assume(last_row >= 0)
        #tl.assume(last_row <= M)

    #tl.assume(last_row <= M)
