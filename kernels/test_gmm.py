
import jax
import jax.numpy as jnp
import gmm as backend  # adjust import to wherever you've defined gmm
import jax_triton as jt

from absl.testing import absltest, parameterized

#common func
from common import (
    generate_inputs,
    ragged_dot_reference,
    TRANS_LHS,
    TRANS_RHS,
    TRANS_OUT,
    TILING,
)


class GroupGemmTest(parameterized.TestCase):
  @parameterized.named_parameters(
      # (test_name, lhs_shape, rhs_shape, group_sizes, tiling, dtype)
      ("test1", (512, 256), (8, 256, 128), [32, 64, 128, 32, 64, 64, 64, 64], (64, 64, 64), jnp.float32),
      ("test2", (512, 256), (8, 256, 128), [32, 64, 128, 32, 64, 64, 64, 64], (64, 64, 64), jnp.bfloat16),
  )
  def test_group_gemm_triton_kernel(
                              self,
                              lhs_shape,
                              rhs_shape,
                              group_sizes,
                              tiling,
                              dtype):   
    M, K = lhs_shape
    G, K2, N = rhs_shape
    self.assertEqual(K, K2, "K dimension must match")

    group_sizes = jnp.array(group_sizes, dtype=jnp.int32)

    self.assertEqual(G, group_sizes.shape[0], "G and group sizes len must match")
    self.assertEqual(M, jnp.sum(group_sizes), "M and sum(group_sizes) must match")

    lhs, rhs = generate_inputs(M, K, N, G,  preferred_element_type=dtype, trans_lhs=False, trans_rhs=False)

    # run gmm
    out = backend.gmm(lhs=lhs, rhs=rhs, group_sizes=group_sizes, tiling=tiling, preferred_element_type=dtype, debug=False)
    out.block_until_ready()
    
    # expected via JAX ragged_dot
    expected = ragged_dot_reference(lhs, rhs, group_sizes)
    
    # shape & dtype sanity
    self.assertEqual(out.shape, (M, N))
    self.assertEqual(out.dtype, dtype)

    # numeric check (allow fp16 tolerance)
    atol = 1e-2 if dtype == jnp.float16 else 1e-3
    self.assertTrue(jnp.allclose(out, expected, atol=atol),
                    msg=f"Mismatch: maxdiff={jnp.max(jnp.abs(out-expected))}")

if __name__ == "__main__":
  absltest.main()
