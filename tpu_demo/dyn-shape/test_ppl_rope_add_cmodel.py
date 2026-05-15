import tilelang
import tilelang.language as T
import torch

N = 64  # must be even for rope_add


def assert_close(name, actual, expected, tol=1e-4):
    max_diff = torch.max(torch.abs(actual.float() - expected.float())).item()
    assert max_diff < tol, f"{name} max diff {max_diff} exceeds tolerance {tol}"
    print(f"{name}: PASSED (max diff {max_diff:.6e})")


M = T.symbolic("M")


@T.prim_func
def dynamic_rope_add(
    X_cos: T.Tensor((M, N), "float32"),
    X_sin: T.Tensor((M, N), "float32"),
    Y: T.Tensor((M, N), "float32"),
):
    with T.Kernel(1, 1, is_cpu=True) as (bx, by):
        local_cos = T.alloc_shared((M, N), "float32")
        local_sin = T.alloc_shared((M, N), "float32")
        x_neg_sin = T.alloc_shared((M, N), "float32")
        local_y = T.alloc_shared((M, N), "float32")

        T.ppl_copy(X_cos[0, 0], local_cos)
        T.ppl_copy(X_sin[0, 0], local_sin)
        T.ppl_mul_C(x_neg_sin, local_sin, T.float32(-1.0))
        T.ppl_rope_add(local_y, local_cos, x_neg_sin, local_cos, local_sin)
        T.ppl_copy(local_y, Y[0, 0])


kernel = tilelang.compile(dynamic_rope_add, out_idx=-1, target="tpu", mode="cmodel")

# Reference: rope_add interleaves even/odd lanes
def rope_ref(x_cos, x_sin):
    out = torch.zeros_like(x_cos)
    out[:, 0::2] = x_cos[:, 0::2] - x_sin[:, 1::2]
    out[:, 1::2] = x_cos[:, 1::2] + x_sin[:, 0::2]
    return out

# Test with M=3
cos1 = torch.randn(3, N).float()
sin1 = torch.randn(3, N).float()
y1 = torch.zeros(3, N).float()
kernel(cos1, sin1, y1)
assert_close("dynamic rope_add M=3", y1, rope_ref(cos1, sin1))

# Test with a different M value to verify dynamic shape works
cos2 = torch.randn(7, N).float()
sin2 = torch.randn(7, N).float()
y2 = torch.zeros(7, N).float()
kernel(cos2, sin2, y2)
assert_close("dynamic rope_add M=7", y2, rope_ref(cos2, sin2))

print("\n" + "=" * 60)
print("ALL PPL.ROPE_ADD TESTS PASSED")
print("=" * 60)
