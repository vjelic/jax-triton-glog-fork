# Python standard library
import math

# JAX
import jax
import jax.numpy as jnp
import jax_triton as jt


# GMM kernel
from triton_tgmm_kernel import (
    triton_tgmm_persistent_kernel_core,
    triton_tgmm_non_persistent_kernel_core,
)


#common func
from common import (
    num_of_cu,
    get_tiling,
    get_tgmm_transposition,
    is_power_of_2,
    next_power_of_2,
    TRANS_LHS,
    TRANS_RHS,
    TRANS_OUT,
    TILING,
)


def get_tgmm_shape(

    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray

) -> tuple[int, int, int, int]:

    assert jnp.ndim(lhs) == 2, f"lhs must have 2 dimensions (it's {jnp.ndim(lhs)})."
    assert jnp.ndim(rhs) == 2, f"rhs must have 2 dimensions (it's {jnp.ndim(rhs)})."
    assert (
        jnp.ndim(group_sizes) == 1
    ), f"group_sizes must have 1 dimension (it's {jnp.ndim(group_sizes)})."

    K, lhs_m = lhs.shape
    rhs_m, N = rhs.shape
    G = group_sizes.shape[0]

    assert (
        lhs_m == rhs_m
    ), f"M dimension of lhs and rhs don't match (lhs = {lhs_m}, rhs = {rhs_m})."
    M = lhs_m

    assert M > 0, f"M must be positive, it's {M}."
    assert K > 0, f"K must be positive, it's {K}."
    assert N > 0, f"N must be positive, it's {N}"
    assert G > 0, f"G must be positive, it's {G}"

    return M, K, N, G

def cdiv(n, d):
    return (n + d - 1) // d

def compute_persistent_grid(
    K: int,
    N: int,
    G: int,
    block_size_k: int,
    block_size_n: int,
    grid_dim: int,
) -> tuple[int]:
    assert K > 0, f"K must be positive, it's {K}."
    assert N > 0, f"N must be positive, it's {N}."
    assert G > 0, f"G must be positive, it's {G}."
    assert is_power_of_2(
        block_size_k
    ), f"K-dimension tile size must be a power of 2 (it's {block_size_k})."
    assert is_power_of_2(
        block_size_n
    ), f"N-dimension tile size must be a power of 2 (it's {block_size_n})."
    assert grid_dim > 0, f"Grid dimension must be positive (it's {grid_dim})."
    num_k_tiles = cdiv(K, block_size_k)
    assert num_k_tiles > 0, f"num_k_tiles must be positive, it's {num_k_tiles}."
    num_n_tiles = cdiv(N, block_size_n)
    assert num_n_tiles > 0, f"num_n_tiles must be positive, it's {num_n_tiles}."
    num_tiles = G * num_k_tiles * num_n_tiles
    assert num_tiles > 0, f"num_tiles must be positive, it's {num_tiles}."
    num_programs = min(grid_dim, num_tiles)
    assert num_programs > 0, f"num_programs must be positive, it's {num_programs}."
    return (num_programs,)


# Triton persistent TGMM implementation.
# ------------------------------------------------------------------------------
def triton_persistent_tgmm(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    preferred_element_type: jnp.dtype = jnp.bfloat16,
    trans_out: bool = False,
    existing_out: jnp.ndarray | None = None,
    tiling: tuple[int, int, int] = TILING,
    autotune: bool = False,
    debug: bool = False, 
) -> jnp.ndarray:

    m, k, n, g = get_tgmm_shape(lhs, rhs, group_sizes)
    block_size_m, block_size_k, block_size_n = get_tiling(m,k,n,tiling)
   
    shape = (g, n, k) if trans_out else (g, k, n)
    out_shape = jax.ShapeDtypeStruct(shape, dtype=preferred_element_type)

    group_size_m=1 # would come from a Lookup Table. [key-value store]: optimization uses
    grid_dim = num_of_cu()    

    print("Running non autotuned persistent TGMM kernel.")
    #trans_lhs, trans_rhs, trans_out, _, _, _ = get_tgmm_transposition(lhs, rhs, out)
    trans_lhs=False

    grid = compute_persistent_grid(
        k,
        n,
        g,
        block_size_k,
        block_size_n,
        grid_dim,
    )

    return  jt.triton_call(
        lhs,
        rhs,
        group_sizes,
        kernel=triton_tgmm_persistent_kernel_core,
        out_shape=out_shape,
        grid=grid,
        num_warps=8,
        num_stages=2,
        #  shapes:
        M=m,
        K=k,
        N=n,
        G=g,
        # Meta-parameters:
        TRANS_LHS=trans_lhs,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_K=block_size_k,
        BLOCK_SIZE_N=block_size_n,
        GROUP_SIZE=group_size_m,
        GRID_DIM=grid_dim,
        debug=debug
    )
        
