
# Python standard library
import argparse
import typing
import math

# JAX
import jax
import jax.numpy as jnp
from jax._src import dtypes

#numpy
import numpy as np

import ctypes


# TODO: Figure out a sensible tiling default.
TILING: tuple[int, int, int] = (64, 64, 64)


# Default transposition.
TRANS_LHS: bool = False
TRANS_RHS: bool = False
TRANS_OUT: bool = False

def is_power_of_2(x: int) -> bool:
    return (x > 0) and (x & (x - 1) == 0)

def next_power_of_2(n: int) -> int:
    if n < 1:
        raise ValueError("n must be >= 1")
    return 2 ** math.ceil(math.log2(n))


def generate_tgmm_inputs(
        M: int,
        K: int,
        N: int,
        preferred_element_type: jnp.dtype = jnp.bfloat16,
        trans_lhs: bool = TRANS_LHS,
        trans_rhs: bool = TRANS_RHS,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
    
    assert M > 0, f"Number of lhs rows M must be positive (M = {M})."
    assert K > 0, f"Number of lhs columns / rhs rows K must be positive (K = {K})."
    assert N > 0, f"Number of rhs columns N must be positive (N = {N})."

    k1, k2 = jax.random.split(jax.random.PRNGKey(0))

    lhs_row, lhs_col = (K, M) if trans_lhs else (M, K)
    rhs_row, rhs_col = (N, K) if trans_rhs else (K, N)

    lhs = jax.random.normal(k1, (lhs_row, lhs_col), dtype=preferred_element_type)
    rhs = jax.random.normal(k2, (rhs_row, rhs_col), dtype=preferred_element_type)
    return lhs, rhs



def generate_inputs(
    M: int,
    K: int,
    N: int,
    G: int,
    preferred_element_type: jnp.dtype = jnp.bfloat16,
    trans_lhs: bool = TRANS_LHS,
    trans_rhs: bool = TRANS_RHS,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:

    

    assert M > 0, f"Number of lhs rows M must be positive (M = {M})."
    assert K > 0, f"Number of lhs columns / rhs rows K must be positive (K = {K})."
    assert N > 0, f"Number of rhs columns N must be positive (N = {N})."
    assert G > 0, f"Number of groups G must be positive (G = {G})."

    k1, k2 = jax.random.split(jax.random.PRNGKey(0))

    lhs_row, lhs_col = (K, M) if trans_lhs else (M, K)
    rhs_row, rhs_col = (N, K) if trans_rhs else (K, N)

    lhs = jax.random.normal(k1, (lhs_row, lhs_col), dtype=preferred_element_type)
    rhs = jax.random.normal(k2, (G, rhs_row, rhs_col), dtype=preferred_element_type)
    return lhs, rhs


def get_tgmm_transposition(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    out: jnp.ndarray
) -> tuple[bool, bool, bool, int, int, int]:
    assert jnp.ndim(lhs) == 2, f"lhs must have 2 dimensions (it's {jnp.ndim(lhs)})."
    assert jnp.ndim(rhs) == 2, f"rhs must have 2 dimensions (it's {jnp.ndim(rhs)})."
    assert jnp.ndim(out) == 3, f"out must have 3 dimensions (it's {jnp.ndim(out)})."

    lhs_k, lhs_m = lhs.shape
    rhs_m, rhs_n = rhs.shape
    G, out_k, out_n = out.shape

    assert (
        lhs_m == rhs_m
    ), f"M dimension of lhs and rhs don't match (lhs = {lhs_m}, rhs = {rhs_m})."
    M = lhs_m
    assert (
        lhs_k == out_k
    ), f"K dimension of lhs and out don't match (lhs = {lhs_k}, rhs = {out_k})."
    K = lhs_k
    assert (
        rhs_n == out_n
    ), f"N dimension of rhs and out don't match (lhs = {rhs_n}, rhs = {out_n})."
    N = rhs_n

    assert M > 0, f"M must be positive, it's {M}."
    assert K > 0, f"K must be positive, it's {K}."
    assert N > 0, f"N must be positive, it's {N}"
    assert G > 0, f"G must be positive, it's {G}"

    is_lhs_row_major = lhs.shape == (M, K)
    is_lhs_col_major = lhs.shape == (K, M)
    assert (
        is_lhs_row_major != is_lhs_col_major
    ), "lhs must be row-major or column-major."

    is_rhs_row_major = rhs.shape == (M, N)
    is_rhs_col_major = rhs.stride() == (N, M)
    assert (
        is_rhs_row_major != is_rhs_col_major
    ), "rhs must be row-major or column-major."

    is_out_row_major = out.shape == (G, K, N)
    is_out_col_major = out.stride() == (G, N, K)
    assert (
        is_out_row_major != is_out_col_major
    ), "out must be row-major or column-major."

    # Get leading dimension according to transposition configuration.
    ld_lhs = M if is_lhs_row_major else K
    ld_rhs = N if is_rhs_row_major else M
    ld_out = N if is_out_row_major else K

    return is_lhs_col_major, is_rhs_col_major, is_out_col_major, ld_lhs, ld_rhs, ld_out

def num_of_cu() -> int:
    '''
    int warp_size;
    if ((status = hipDeviceGetAttribute(&warp_size,
      hipDeviceAttributeWarpSize, dev)) != hipSuccess) {
      return status;
    }
    '''
    # JAX device and get its numeric device‐ID
    device = jax.devices()[0]
    dev_id = device.id

    # load the ROCm (HIP) runtime
    hip = ctypes.cdll.LoadLibrary("libamdhip64.so")

    # enum for "multiprocessor count" in hip_runtime_api.h
    #    – hipDeviceAttributeMultiprocessorCount == 63
    HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 63

    # 4) call hipDeviceGetAttribute
    cu_count = ctypes.c_int()
    err = hip.hipDeviceGetAttribute(
        ctypes.byref(cu_count),
        HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        ctypes.c_int(dev_id),
    )
    if err != 0:
        raise RuntimeError(f"HIP call failed with error code {err}")

    print(f"Device #{dev_id} ({device.device_kind!r}):")
    print(f"  Compute Units:        {cu_count.value}")

    return cu_count.value


def get_wrap_size()-> int:
    '''
    int warp_size;
    if ((status = hipDeviceGetAttribute(&warp_size,
      hipDeviceAttributeWarpSize, dev)) != hipSuccess) {
      return status;
    }
    '''
    device = jax.devices()[0]
    dev_id = device.id

    # load the ROCm (HIP) runtime
    hip = ctypes.cdll.LoadLibrary("libamdhip64.so")

    # 3) the enum for "wrap size" in hip_runtime_api.h
    #    – hipDeviceAttributeWrapSize = 87
    HIP_DEVICE_ATTRIBUTE_WRAP_SIZE = 87

    # 4) call hipDeviceGetAttribute
    warp_size = ctypes.c_int()
    err = hip.hipDeviceGetAttribute(
        ctypes.byref(warp_size),
        HIP_DEVICE_ATTRIBUTE_WRAP_SIZE,
        ctypes.c_int(dev_id),
    )
    if err != 0:
        raise RuntimeError(f"HIP call failed with error code {err}")

    print(f"Device #{dev_id} ({device.device_kind!r}):")
    print(f"  wrap size:        {warp_size.value}")

    return warp_size.value


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


def ragged_dot_reference(
    lhs,
    rhs,
    group_sizes,
) -> np.array:
  """Reference ragged dot implementation."""
  m, lk = lhs.shape
  group_count, rk, n = rhs.shape
  assert lk == rk
  assert group_count == group_sizes.shape[0]
  assert lhs.dtype == rhs.dtype

  out = np.zeros((m, n), dtype=lhs.dtype)
  result_iota = np.expand_dims(np.arange(out.shape[0]), list(range(1, out.ndim)))
  start = 0
  for i, size in enumerate(group_sizes):
    out += np.where(
        np.logical_and(start <= result_iota, result_iota < (start + size)),
        np.einsum(
          "nk,km->nm",
          lhs,
          rhs[i, :, :],
          dtype=np.float32 if lhs.dtype == dtypes.bfloat16 else np.float32,
        ),
        np.zeros(out.shape, dtype=out.dtype),
    )
    start += size
  return out.astype(dtypes.bfloat16) if lhs.dtype == dtypes.bfloat16 else out