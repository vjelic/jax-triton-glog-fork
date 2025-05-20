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