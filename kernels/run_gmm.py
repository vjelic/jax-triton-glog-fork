
import jax
import jax.numpy as jnp
import gmm as backend  # adjust import to wherever you've defined gmm

import functools
import timeit


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
    
    preferred_element_type=jnp.bfloat16 #jnp.float32
    precision_ = "F32_F32_F32" if preferred_element_type==jnp.float32 else "BF16_BF16_F32"
    print(f"precision_ = {precision_}")

    _perf = False
    if _perf:
        M, K, N = 393216, 2048, 1408
        group_sizes = jnp.array([49152,65536,32768,34816,63488,49152,32768,65536], dtype=jnp.int32)    
    else:
        M, N, K = 512, 256, 128
        group_sizes = jnp.array([32, 64, 128, 32, 64, 64, 64, 64], dtype=jnp.int32)

    G = group_sizes.shape[0] 
    lhs, rhs = generate_inputs(M, K, N, G,  preferred_element_type=preferred_element_type, trans_lhs=False, trans_rhs=False)        
    tiling=TILING

    print(f"lhs shape={lhs.shape}, rhs shape={rhs.shape}, preferred_element_type={preferred_element_type}")

    func_group_gmm = functools.partial(backend.gmm, lhs, rhs, group_sizes, tiling, preferred_element_type, False,)
    jax_ragged_dot = functools.partial(jax.lax.ragged_dot, lhs, rhs, group_sizes=group_sizes,precision=precision_, )

    if _perf:

        n=50
        grp_gemm_triton = func_group_gmm().block_until_ready() # warm up
        t = timeit.timeit(lambda: func_group_gmm().block_until_ready(), number=n)
        #t2 = timeit.timeit(lambda: jax_ragged_dot().block_until_ready(), number=n)
        print(f"Average triton group gemm {n} runs: {t/n:.6f} s")
        #print(f"Average jax_ragged_dot {n} runs: {t2/n:.6f} s")

    else:

        grp_gemm_triton = func_group_gmm().block_until_ready()
        ragged_dot      = jax.lax.ragged_dot(lhs, rhs, group_sizes=group_sizes,precision=precision_).block_until_ready()
        ragged_dot_ref  = ragged_dot_reference(lhs, rhs, group_sizes=group_sizes)

        atol = 5e-2 if preferred_element_type == jnp.bfloat16 else 5e-3
        if not jnp.allclose(ragged_dot, ragged_dot_ref, atol):
            diff = jnp.abs(ragged_dot - ragged_dot_ref).max()
            print(f"\nWARNING: ragged_dot and ragged_dot_ref result mismatch; Max-diff={diff}\n")
        else:
            print("Test passed: ragged_dot & ragged_dot_ref results matched\n")

        if not jnp.allclose(ragged_dot_ref, grp_gemm_triton, atol):
            diff = jnp.abs(ragged_dot_ref - grp_gemm_triton).max()
            raise ValueError(
                f"Mismatch between grp_gemm_triton and ragged_dot_ref. Max diff={diff}\n"
                f" grp_gemm_triton  = {grp_gemm_triton}\n\n ragged_dot_ref = {ragged_dot_ref}"
                )
        else:
            print("Test passed: grp_gemm_triton & ragged_dot_ref results matched")
    


if __name__ == "__main__":
    from absl import app
    app.run(main)