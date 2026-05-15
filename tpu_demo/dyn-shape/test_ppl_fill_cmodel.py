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
def dynamic_fill(
    Y: T.Tensor((M, N), "float32"),
):
    with T.Kernel(1, 1, is_cpu=True) as (bx, by):
        local = T.alloc_shared((M, N), "float32")
        T.ppl_fill(local, T.float32(42.0))
        T.ppl_copy(local, Y[0, 0])


kernel = tilelang.compile(dynamic_fill, out_idx=-1, target="tpu", mode="cmodel")

# Test with M=3
y1 = torch.zeros(3, N).float()
kernel(y1)
assert_close("dynamic fill M=3", y1, torch.full((3, N), 42.0).float(), tol=1e-6)

# Test with a different M value to verify dynamic shape works
y2 = torch.zeros(7, N).float()
kernel(y2)
assert_close("dynamic fill M=7", y2, torch.full((7, N), 42.0).float(), tol=1e-6)

print("\n" + "=" * 60)
print("ALL PPL.FILL TESTS PASSED")
print("=" * 60)
