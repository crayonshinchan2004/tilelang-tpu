import tilelang
import tilelang.language as T
import torch

N = 64


def assert_close(name, actual, expected, tol=1e-4):
    max_diff = torch.max(torch.abs(actual.float() - expected.float())).item()
    assert max_diff < tol, f"{name} max diff {max_diff} exceeds tolerance {tol}"
    print(f"{name}: PASSED (max diff {max_diff:.6e})")


M = T.symbolic("M")


@T.prim_func
def dynamic_div(
    X: T.Tensor((M, N), "float32"),
    Y: T.Tensor((M, N), "float32"),
    Z: T.Tensor((M, N), "float32"),
):
    with T.Kernel(1, 1, is_cpu=True) as (bx, by):
        local_x = T.alloc_shared((M, N), "float32")
        local_y = T.alloc_shared((M, N), "float32")
        local_z = T.alloc_shared((M, N), "float32")

        T.ppl_copy(X[0, 0], local_x)
        T.ppl_copy(Y[0, 0], local_y)
        T.ppl_div(local_z, local_x, local_y)
        T.ppl_copy(local_z, Z[0, 0])


kernel = tilelang.compile(dynamic_div, out_idx=-1, target="tpu", mode="cmodel")

# Test with M=3
x1 = torch.randn(3, N).float()
y1 = torch.randn(3, N).float() + 1.0  # avoid division by zero
z1 = torch.zeros(3, N).float()
kernel(x1, y1, z1)
assert_close("dynamic div M=3", z1, x1 / y1)

# Test with a different M value
x2 = torch.randn(7, N).float()
y2 = torch.randn(7, N).float() + 1.0
z2 = torch.zeros(7, N).float()
kernel(x2, y2, z2)
assert_close("dynamic div M=7", z2, x2 / y2)

print("\n" + "=" * 60)
print("ALL PPL.DIV TESTS PASSED")
print("=" * 60)
