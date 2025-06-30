

"""Grouped matrix multiplication operations with custom VJPs."""

import jax
import gmm as backend
import tgmm as backend_tgmm
import jax.numpy as jnp


gmm = jax.custom_vjp(
    backend.gmm,
    nondiff_argnums=(3, 4, 5),
)


def _gmm_fwd(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    tiling: tuple[int, int, int] = (128, 128, 128),
    preferred_element_type: jnp.dtype = jnp.bfloat16,
    debug: bool = False,

    ) -> tuple[
        jnp.ndarray,
        tuple[
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            tuple[int, int, int],
            jnp.dtype,
            ],
    ]:
  """Forward function for GMM VJP."""
  out = backend.gmm(
      lhs,
      rhs,
      group_sizes,
      tiling,
      preferred_element_type,
      debug,      
  )
  return out, (lhs, rhs, group_sizes, tiling, preferred_element_type)


def _gmm_bwd(
        residual: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, tuple[int, int, int], jnp.dtype, bool,],
        grad: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, None, None, jnp.ndarray]:

    """Backward function for throughput GMM VJP.""" 

    lhs, rhs, group_sizes, tiling, preferred_element_type, debug = residual

    grad_lhs = backend.gmm(
                                    grad,
                                    rhs,
                                    group_sizes,
                                    tiling,
                                    preferred_element_type
                                    )

    grad_rhs = backend_tgmm.triton_persistent_tgmm(
                                    lhs,
                                    grad,
                                    group_sizes,
                                    tiling,
                                    preferred_element_type,
                                    False,
                                    None,
                                    False,
                                    debug,
                                    )

    # NOTE: If the rhs transposition is fused into the forward pass we need to
    # return the transpose of the rhs gradient that we calculated above.
    #
    # TODO(tgale, enriqueps, apaske): Fuse this transposition into the tgmm.
    # grad_rhs = grad_rhs.swapaxes(1, 2) if transpose_rhs else grad_rhs
    return grad_lhs, grad_rhs, None, None, grad


gmm.defvjp(_gmm_fwd, _gmm_bwd)