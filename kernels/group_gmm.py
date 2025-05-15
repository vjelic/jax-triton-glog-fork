# -*- coding: utf-8 -*-


# GMM problem description:
# * Input tensors:
#   * lhs is (M, K) bf16
#   * rhs is (G, K, N) bf16
#   * group_sizes is (G,) int32
# * Output tensors:
#   * out is (M, N) bf16


# Imports.
# ------------------------------------------------------------------------------

# Python standard library
import argparse
import typing
import math

# JAX
import jax
import jax.numpy as jnp
import jax_triton as jt

#common func
from common import (
    generate_inputs,
    get_num_cores,
    ragged_dot_reference,
    TRANS_LHS,
    TRANS_RHS,
    TRANS_OUT,
    TILING,
)

# GMM kernel
from gmm_kernel import triton_gmm_kernel_core

# pytest
import pytest

# Parameter checking functions.
# ------------------------------------------------------------------------------

def shape_from_input(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int]:

    assert jnp.ndim(lhs) == 2, f"lhs must have 2 dimensions (it's {jnp.ndim(lhs)})."
    assert jnp.ndim(rhs) == 3, f"rhs must have 3 dimensions (it's {jnp.ndim(rhs)})."
    assert (
        jnp.ndim(group_sizes) == 1
    ), f"group_sizes must have 1 dimension (it's {jnp.ndim(group_sizes)})."

    M, lhs_k = lhs.shape
    K = lhs_k
    rhs_g, rhs_k, N = rhs.shape
    G = rhs_g
    group_sizes_g = group_sizes.shape[0]

    assert (
        lhs_k == rhs_k
    ), f"K dimension of lhs and rhs don't match (lhs = {lhs_k}, rhs = {rhs_k})."
    
    assert (
        rhs_g == group_sizes_g
    ), f"G dimension of rhs and group_sizes don't match (rhs = {rhs_g}, group_sizes = {group_sizes_g})."
    

    assert M > 0, f"M must be positive, it's {M}."
    assert K > 0, f"K must be positive, it's {K}."
    assert N > 0, f"N must be positive, it's {N}"
    assert G > 0, f"G must be positive, it's {G}"

    return M, K, N, G


###########################################################################

def is_power_of_2(x: int) -> bool:
    return (x > 0) and (x & (x - 1) == 0)

def next_power_of_2(n: int) -> int:
    if n < 1:
        raise ValueError("n must be >= 1")
    return 2 ** math.ceil(math.log2(n))


def get_tiling(
    M: int,
    K: int,
    N: int,
    tiling: tuple[int, int, int]
    ) -> tuple[int, int, int]:
    
    """
    Compute and validate the tile sizes for a GEMM‑style operation, clamped to the next power‑of‑2.

    Given desired maximum tile dimensions in `tiling`, this function picks the minimum of each
    desired size and the next power‑of‑2 of the corresponding matrix dimension, then ensures
    each resulting tile size is itself a power of 2.

    Args:
        M (int): Number of rows in the left-hand matrix; must be > 0.
        K (int): Shared inner dimension (cols of lhs / rows of rhs); must be > 0.
        N (int): Number of columns in the right-hand matrix; must be > 0.
        tiling (tuple[int, int, int]): Desired (max) tile sizes for the M, K, and N dimensions.

    Returns:
        tuple[int, int, int]: A triple `(block_size_m, block_size_k, block_size_n)` where
            each block size is the minimum of the provided tiling and the next power of two
            of the corresponding dimension, and is guaranteed to be a power of two.

    Raises:
        AssertionError: If any of `M`, `K`, or `N` is non‑positive, if `tiling` does not have
            exactly three elements, or if any computed block size is not a power of two.
    """

    assert M > 0, f"Number of lhs rows M must be positive (M = {M})."
    assert K > 0, f"Number of lhs columns / rhs rows K must be positive (K = {K})."
    assert N > 0, f"Number of rhs columns N must be positive (N = {N})."
    assert len(tiling) == 3, f"tiling must have 3 dimensions (it's = {len(tiling)})."

    block_size_m, block_size_k, block_size_n = tiling

    # Pick smaller block sizes for toy shapes.
    block_size_m = min(next_power_of_2(M), block_size_m)
    block_size_k = min(next_power_of_2(K), block_size_k)
    block_size_n = min(next_power_of_2(N), block_size_n)
    
    assert is_power_of_2(
        block_size_m
    ), f"M-dimension tile size must be a power of 2 (it's {block_size_m})."
    assert is_power_of_2(
        block_size_k
    ), f"K-dimension tile size must be a power of 2 (it's {block_size_k})."
    assert is_power_of_2(
        block_size_n
    ), f"N-dimension tile size must be a power of 2 (it's {block_size_n})."

    return block_size_m, block_size_k, block_size_n


################################################################################################

def num_gpus() -> int:
    devices = jax.devices()
    print("All devices:", devices)
    num_gpus = sum(1 for d in devices if d.platform == "gpu")
    print("Number of GPUs:", num_gpus)
    return num_gpus

def cdiv(n, d):
    return (n + d - 1) // d

def compute_grid(
    N: int,
    block_size_m: int,
    block_size_n: int,
    group_sizes: jnp.ndarray,
) -> tuple[int]:
    assert N > 0, f"N must be positive, it's {N}."
    assert is_power_of_2(
        block_size_m
    ), f"M-dimension tile size must be a power of 2 (it's {block_size_m})."
    assert is_power_of_2(
        block_size_n
    ), f"N-dimension tile size must be a power of 2 (it's {block_size_n})."
    assert bool(jnp.all(group_sizes >= 0)), "All group_sizes must be non-negative."
    num_m_tiles = (group_sizes + block_size_m - 1) // block_size_m
    assert bool(jnp.all(num_m_tiles >= 0)), "All num_m_tiles must be non-negative."
    num_n_tiles = cdiv(N, block_size_n)
    assert num_n_tiles > 0, f"num_n_tiles must be positive, it's {num_n_tiles}."
    num_tiles = int(jnp.sum(num_m_tiles * num_n_tiles))
    assert num_tiles > 0, f"num_tiles must be positive, it's {num_tiles}."
    num_programs = int(min(get_num_cores(), num_tiles))
    assert num_programs > 0, f"num_programs must be positive, it's {num_programs}."
    print(f"num_programs={num_programs}")
    return (num_programs,)

### adding jax-triton wrapper for group_gemm kernel ###

def group_gemm(lhs: jnp.ndarray,
               rhs: jnp.ndarray,
               group_sizes: jnp.ndarray,
               tiling: tuple[int, int, int] = TILING,
               preferred_element_type: jnp.dtype = jnp.bfloat16,
               debug: bool = False, 
    ) -> jnp.ndarray:

    #check_input_device_dtype(lhs, rhs, group_sizes)
    m, k, n, g = shape_from_input(lhs, rhs, group_sizes) #TODO need to check with transpose case..

    block_size_m, block_size_k, block_size_n = get_tiling(m,k,n,tiling)

    out_shape = jax.ShapeDtypeStruct(shape=(m, n), dtype=preferred_element_type)    
   
    grid = compute_grid(n, block_size_m, block_size_n, group_sizes)
    is_k_divisible_by_block_k = (k%block_size_k)==0 
    print(f"is_k_divisible_by_block_k={is_k_divisible_by_block_k}")

    group_size_m=1 # would come from a Lookup Table. [key-value store]: optimization uses
    
    return  jt.triton_call(
        lhs,
        rhs,
        group_sizes,
        kernel=triton_gmm_kernel_core,
        out_shape=out_shape,
        grid=grid,
        num_warps=8,
        num_stages=2,
        #  shapes:
        M=m,
        K=k,
        N=n,
        G=g,
        #  strides:
        stride_lhs_m=k,
        stride_lhs_k=1,
        stride_rhs_g=k*n,
        stride_rhs_k=n,
        stride_rhs_n=1,
        stride_out_m=n,
        stride_out_n=1,
        # Meta-parameters:
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_K=block_size_k,
        BLOCK_SIZE_N=block_size_n,
        K_DIVISIBLE_BY_BLOCK_SIZE_K=is_k_divisible_by_block_k,
        GROUP_SIZE_M=group_size_m,
        debug=debug
    )
