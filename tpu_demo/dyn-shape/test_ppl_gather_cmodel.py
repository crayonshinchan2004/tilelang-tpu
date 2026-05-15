import tilelang
import tilelang.language as T
import torch

HIDDEN_SIZE = 64
NUM_INDICES = 8


def assert_close(name, actual, expected, tol=1e-5):
    max_diff = torch.max(torch.abs(actual.float() - expected.float())).item()
    assert max_diff < tol, f"{name} max diff {max_diff} exceeds tolerance {tol}"
    print(f"{name}: PASSED (max diff {max_diff:.6e})")


M = T.symbolic("M")


@T.prim_func
def dynamic_gather(
    Param: T.Tensor((M, HIDDEN_SIZE), "float32"),
    Index: T.Tensor((NUM_INDICES, 1), "uint32"),
    Output: T.Tensor((NUM_INDICES, HIDDEN_SIZE), "float32"),
):
    with T.Kernel(1, 1, is_cpu=True) as (bx, by):
        T.ppl_gather(Output, Param, Index, M)


kernel = tilelang.compile(dynamic_gather, out_idx=-1, target="tpu", mode="cmodel")

# Test with M=128
param1 = torch.randn(128, HIDDEN_SIZE).float()
idx1 = torch.randint(0, 128, (NUM_INDICES, 1), dtype=torch.uint32)
out1 = torch.zeros(NUM_INDICES, HIDDEN_SIZE).float()
kernel(param1, idx1, out1)
assert_close("dynamic gather M=128", out1, param1[idx1.view(-1).long()])

# Test with a different M value
param2 = torch.randn(64, HIDDEN_SIZE).float()
idx2 = torch.randint(0, 64, (NUM_INDICES, 1), dtype=torch.uint32)
out2 = torch.zeros(NUM_INDICES, HIDDEN_SIZE).float()
kernel(param2, idx2, out2)
assert_close("dynamic gather M=64", out2, param2[idx2.view(-1).long()])

print("\n" + "=" * 60)
print("ALL PPL.GATHER TESTS PASSED")
print("=" * 60)
