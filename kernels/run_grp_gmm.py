
import jax
import jax.numpy as jnp
import group_gmm as ggmm  # adjust import to wherever you've defined group_gemm

#common func
from common import (
    generate_inputs,
    ragged_dot_reference,
    TRANS_LHS,
    TRANS_RHS,
    TRANS_OUT,
    TILING,
)

def main(unused_argv):

    preferred_element_type=jnp.bfloat16
    #preferred_element_type=jnp.float32
    precision_ = "F32_F32_F32" if preferred_element_type==jnp.float32 else "BF16_BF16_F32"
    print(f"precision_ = {precision_}")

    M, N, K = 512, 256, 128
    group_sizes = jnp.array([32, 64, 128, 32, 64, 64, 64, 64], dtype=jnp.int32)
    G = group_sizes.shape[0]
 
    lhs, rhs = generate_inputs(M, K, N, G,  preferred_element_type=preferred_element_type, trans_lhs=False, trans_rhs=False)        
    
    tiling=TILING
    
    grp_gemm_triton = ggmm.group_gemm(lhs, rhs, group_sizes, tiling, preferred_element_type, debug=False)
    ragged_dot      = jax.lax.ragged_dot(lhs, rhs, group_sizes=group_sizes,precision=precision_)
    ragged_dot_ref  = ragged_dot_reference(lhs, rhs, group_sizes=group_sizes)
   
    grp_gemm_triton.block_until_ready()
    ragged_dot.block_until_ready()
    

    atol = 1e-2 if preferred_element_type == jnp.float16 else 1e-3
    if not jnp.allclose(ragged_dot, ragged_dot_ref, atol):
        diff = jnp.abs(ragged_dot - ragged_dot_ref).max()
        print(f"\nragged_dot and ragged_dot_ref result mismatch; Max-diff={diff}\n")
    else:
        print("\nragged_dot & ragged_dot_ref results matched\n")


    if not jnp.allclose(ragged_dot_ref, grp_gemm_triton, atol):
        diff = jnp.abs(ragged_dot_ref - grp_gemm_triton).max()
        raise ValueError(
            f"Mismatch between grp_gemm_triton and ragged_dot_ref. Max diff={diff}\n"
            f" grp_gemm_triton  = {grp_gemm_triton}\n\n ragged_dot_ref = {ragged_dot_ref}"
        )
    else:
        print("grp_gemm_triton & ragged_dot_ref results matched")
    


if __name__ == "__main__":
    from absl import app
    app.run(main)