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

# Triton
import triton
import triton.language as tl

# JAX
import jax
import jax.numpy as jnp
import jax_triton as jt

# GMM kernel
from gmm_kernel import triton_gmm_kernel_core

# pytest
import pytest

# TODO: Figure out a sensible tiling default.
TILING: tuple[int, int, int] = (32, 32, 32)

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


def is_power_of_2(x: int) -> bool:
    return (x > 0) and (x & (x - 1) == 0)


def check_tiling(tiling: tuple[int, int, int]) -> tuple[int, int, int]:
    assert len(tiling) == 3, f"tiling must have 3 dimensions (it's = {len(tiling)})."
    block_size_m, block_size_k, block_size_n = tiling
    assert is_power_of_2(
        block_size_m
    ), f"M-dimension tile size must be a power of 2 (it's {block_size_m})."
    assert is_power_of_2(
        block_size_k
    ), f"K-dimension tile size must be a power of 2 (it's {block_size_k})."
    assert is_power_of_2(
        block_size_n
    ), f"N-dimension tile size must be a power of 2 (it's {block_size_n})."
    return tiling





def num_gpus() -> int:
    devices = jax.devices()
    print("All devices:", devices)
    num_gpus = sum(1 for d in devices if d.platform == "gpu")
    print("Number of GPUs:", num_gpus)
    return num_gpus


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
    num_n_tiles = triton.cdiv(N, block_size_n)
    assert num_n_tiles > 0, f"num_n_tiles must be positive, it's {num_n_tiles}."
    num_tiles = int(jnp.sum(num_m_tiles * num_n_tiles))
    assert num_tiles > 0, f"num_tiles must be positive, it's {num_tiles}."
    num_programs = int(min(num_gpus(), num_tiles))
    assert num_programs > 0, f"num_programs must be positive, it's {num_programs}."
    return (num_programs,)

### adding jax-triton wrapper for group_gemm kernel ###

def group_gemm(lhs: jnp.ndarray,
               rhs: jnp.ndarray,
               group_sizes,
               tiling: tuple[int, int, int] = TILING,
               preferred_element_type: jnp.dtype = jnp.bfloat16,
               existing_out: jnp.ndarray | None = None,
               debug: bool = False, 
    ) -> jnp.ndarray:

    #check_input_device_dtype(lhs, rhs, group_sizes)
    m, k, n, g = shape_from_input(lhs, rhs, group_sizes)
    block_size_m, block_size_k, block_size_n = check_tiling(tiling)
    out_shape = jax.ShapeDtypeStruct(shape=(m, n), dtype=lhs.dtype)    
    out_ = jnp.zeros((m, n), dtype=preferred_element_type)

    grid = compute_grid(n, block_size_m, block_size_n, group_sizes)
    
    return jt.triton_call(
        lhs,
        rhs,
        group_sizes,
        out_,
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
        K_DIVISIBLE_BY_BLOCK_SIZE_K=1,
        debug=debug
    )




def main(unused_argv):

    preferred_element_type=jnp.bfloat16

    group_sizes = jnp.array([32, 64, 128, 32, 64, 64, 64, 64], dtype=jnp.int32)
    g=group_sizes.shape[0]
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    lhs = jax.random.normal(k1, (512, 512), dtype=preferred_element_type)
    rhs = jax.random.normal(k2, (g, 512, 512), dtype=preferred_element_type)

    tiling=TILING
    

    out_grp_gemm = group_gemm(lhs, rhs, group_sizes, tiling, preferred_element_type, debug=False)
    out_ragged = jax.lax.ragged_dot(lhs, rhs, group_sizes=group_sizes)

    if not jnp.allclose(out_grp_gemm, out_ragged, 1e-3):
        diff = jnp.abs(out_grp_gemm - out_ragged).max()
        raise ValueError(
            f"Mismatch between gmm and ragged_dot. Max diff={diff}\n"
            f"out_gmm = {out_grp_gemm}\n\nout_ragged = {out_ragged}"
        )


if __name__ == "__main__":
    from absl import app
    app.run(main)