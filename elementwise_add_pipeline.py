import argparse
import os
import sys

TL_ROOT="/home/developer/workspace/git/github/tile-ai/tilelang-ascend"
os.environ['TL_ROOT'] = f"{TL_ROOT}"
os.environ['PYTHONPATH'] = f"{TL_ROOT}:{os.environ['PYTHONPATH']}"
os.environ['ACL_OP_INIT_MODE'] = "1"
sys.path.append(f"{TL_ROOT}")

import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--m", type=int, default=1024, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension")
args = parser.parse_args()

M = args.m
N = args.n

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True
}

@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def vec_add(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    block_NN = 32

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_NN), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_NN), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_NN), dtype)
            with T.Scope("V"):
                for n_i in T.Pipelined(T.ceildiv(block_N, block_NN), num_stages=2):
                    T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N + n_i * block_NN], a_ub)
                    T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N + n_i * block_NN], b_ub)

                    # T.barrier_all()
                    T.tile.add(c_ub, a_ub, b_ub)
                    # T.barrier_all()

                    T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N + n_i * block_NN])

    return main


func = vec_add(M, N, 128, 256)

torch.manual_seed(0)

a = torch.randn(M, N).npu()
b = torch.randn(M, N).npu()

torch.npu.synchronize()
print("init successful!")

c = func(a, b)

ref_c = a + b

torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
print("Kernel Output Match!")
