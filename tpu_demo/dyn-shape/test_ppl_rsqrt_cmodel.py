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
def dynamic_rsqrt(
    X: T.Tensor((M, N), "float32"),
    Y: T.Tensor((M, N), "float32"),
):
    with T.Kernel(1, 1, is_cpu=True) as (bx, by):
        local_x = T.alloc_shared((M, N), "float32")
        local_y = T.alloc_shared((M, N), "float32")

        T.ppl_copy(X[0, 0], local_x)
        T.ppl_rsqrt(local_y, local_x)
        T.ppl_copy(local_y, Y[0, 0])


kernel = tilelang.compile(dynamic_rsqrt, out_idx=-1, target="tpu", mode="cmodel")

# Test with M=3
x1 = torch.randn(3, N).float().abs() + 0.1  # positive inputs for rsqrt
y1 = torch.zeros(3, N).float()
kernel(x1, y1)
assert_close("dynamic rsqrt M=3", y1, torch.rsqrt(x1))

# Test with a different M value to verify dynamic shape works
x2 = torch.randn(7, N).float().abs() + 0.1
y2 = torch.zeros(7, N).float()
kernel(x2, y2)
assert_close("dynamic rsqrt M=7", y2, torch.rsqrt(x2))

print("\n" + "=" * 60)
print("ALL PPL.RSQRT TESTS PASSED")
print("=" * 60)
