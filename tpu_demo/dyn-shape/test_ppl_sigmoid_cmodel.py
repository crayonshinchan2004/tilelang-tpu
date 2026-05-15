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
def dynamic_sigmoid(
    X: T.Tensor((M, N), "float32"),
    Y: T.Tensor((M, N), "float32"),
):
    with T.Kernel(1, 1, is_cpu=True) as (bx, by):
        local_x = T.alloc_shared((M, N), "float32")
        local_y = T.alloc_shared((M, N), "float32")
        work0 = T.alloc_shared((M, N), "float32")
        work1 = T.alloc_shared((M, N), "float32")
        coeff = T.alloc_shared([64, 32], "float32")
        table = T.alloc_shared([64, 192], "float32")

        T.ppl_copy(X[0, 0], local_x)
        T.ppl_sigmoid(local_y, local_x, work0, work1, coeff, table)
        T.ppl_copy(local_y, Y[0, 0])


kernel = tilelang.compile(dynamic_sigmoid, out_idx=-1, target="tpu", mode="cmodel")

# Test with M=3
x1 = torch.randn(3, N).float()
y1 = torch.zeros(3, N).float()
kernel(x1, y1)
assert_close("dynamic sigmoid M=3", y1, torch.sigmoid(x1))

# Test with a different M value to verify dynamic shape works
x2 = torch.randn(7, N).float()
y2 = torch.zeros(7, N).float()
kernel(x2, y2)
assert_close("dynamic sigmoid M=7", y2, torch.sigmoid(x2))

print("\n" + "=" * 60)
print("ALL PPL.SIGMOID TESTS PASSED")
print("=" * 60)
