import tilelang
import tilelang.language as T
import torch

K = 32
N = 32


def assert_close(name, actual, expected, tol=1e-2):
    max_diff = torch.max(torch.abs(actual.float() - expected.float())).item()
    assert max_diff < tol, f"{name} max diff {max_diff} exceeds tolerance {tol}"
    print(f"{name}: PASSED (max diff {max_diff:.6e})")


M = T.symbolic("M")


@T.prim_func
def dynamic_gemm(
    A: T.Tensor((M, K), "float16"),
    B: T.Tensor((K, N), "float16"),
    C: T.Tensor((M, N), "float32"),
):
    with T.Kernel(1, 1, is_cpu=True) as (bx, by):
        A_shared = T.alloc_shared((M, K), "float16")
        B_shared = T.alloc_shared((K, N), "float16")
        C_shared = T.alloc_shared((M, N), "float32")

        T.ppl_fill(C_shared, T.float32(0))
        T.ppl_copy(A[0, 0], A_shared)
        T.ppl_copy(B[0, 0], B_shared)
        T.ppl_gemm(A_shared, B_shared, C_shared)
        T.ppl_copy(C_shared, C[0, 0])


kernel = tilelang.compile(dynamic_gemm, out_idx=-1, target="tpu", mode="cmodel")

# Test with M=16
a1 = torch.randn(16, K).half()
b1 = torch.randn(K, N).half()
c1 = torch.zeros(16, N).float()
kernel(a1, b1, c1)
assert_close("dynamic gemm M=16", c1, torch.matmul(a1, b1).float())

# Test with a different M value to verify dynamic shape works
a2 = torch.randn(24, K).half()
b2 = torch.randn(K, N).half()
c2 = torch.zeros(24, N).float()
kernel(a2, b2, c2)
assert_close("dynamic gemm M=24", c2, torch.matmul(a2, b2).float())

print("\n" + "=" * 60)
print("ALL PPL.GEMM TESTS PASSED")
print("=" * 60)
