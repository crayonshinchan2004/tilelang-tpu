import tilelang
import tilelang.language as T
import torch

N = 64


def assert_close(name, actual, expected, tol=1e-2):
    max_diff = torch.max(torch.abs(actual.float() - expected.float())).item()
    assert max_diff < tol, f"{name} max diff {max_diff} exceeds tolerance {tol}"
    print(f"{name}: PASSED (max diff {max_diff:.6e})")


M = T.symbolic("M")


@T.prim_func
def dynamic_reduce_max(
    X: T.Tensor((M, N), "float16"),
    Y: T.Tensor((M, 1), "float16"),
):
    with T.Kernel(1, 1, is_cpu=True) as (bx, by):
        X_shared = T.alloc_shared((M, N), "float16")
        Y_shared = T.alloc_shared((M, 1), "float16")

        T.ppl_copy(X[0, 0], X_shared)
        T.ppl_reduce_max(X_shared, Y_shared, 1)
        T.ppl_copy(Y_shared, Y[0, 0])


kernel = tilelang.compile(dynamic_reduce_max, out_idx=-1, target="tpu", mode="cmodel")

# Test with M=3
x1 = torch.randn(3, N).half()
y1 = torch.zeros(3, 1).half()
kernel(x1, y1)
assert_close("dynamic reduce_max M=3", y1, torch.max(x1, dim=1, keepdim=True).values)

# Test with a different M value to verify dynamic shape works
x2 = torch.randn(7, N).half()
y2 = torch.zeros(7, 1).half()
kernel(x2, y2)
assert_close("dynamic reduce_max M=7", y2, torch.max(x2, dim=1, keepdim=True).values)

print("\n" + "=" * 60)
print("ALL PPL.REDUCE_MAX TESTS PASSED")
print("=" * 60)
