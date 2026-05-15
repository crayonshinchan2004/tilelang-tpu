import tilelang
import tilelang.language as T
import torch

K = 8


def assert_close(name, actual, expected, tol=1e-4):
    max_diff = torch.max(torch.abs(actual.float() - expected.float())).item()
    assert max_diff < tol, f"{name} max diff {max_diff} exceeds tolerance {tol}"
    print(f"{name}: PASSED (max diff {max_diff:.6e})")


M = T.symbolic("M")


@T.prim_func
def dynamic_topk(
    Input: T.Tensor((M,), "float32"),
    Output: T.Tensor((M,), "float32"),
    Indices: T.Tensor((M,), "int32"),
):
    with T.Kernel(1, 1, is_cpu=True) as (bx, by):
        T.ppl_topk(Output, Indices, Input, K, 1, M)


kernel = tilelang.compile(dynamic_topk, out_idx=[1, 2], target="tpu", mode="cmodel")

# Test with M=128
x1 = torch.randn(128).float()
out1 = torch.zeros(128).float()
idx1 = torch.zeros(128, dtype=torch.int32)
kernel(x1, out1, idx1)
ref_vals1, ref_idx1 = torch.topk(x1, K, largest=True)
assert_close("topk value M=128", out1[:K], ref_vals1)
assert_close("topk index M=128", idx1[:K].float(), ref_idx1.float())

# Test with a different M value
x2 = torch.randn(256).float()
out2 = torch.zeros(256).float()
idx2 = torch.zeros(256, dtype=torch.int32)
kernel(x2, out2, idx2)
ref_vals2, ref_idx2 = torch.topk(x2, K, largest=True)
assert_close("topk value M=256", out2[:K], ref_vals2)
assert_close("topk index M=256", idx2[:K].float(), ref_idx2.float())

print("\n" + "=" * 60)
print("ALL PPL.TOPK TESTS PASSED")
print("=" * 60)
