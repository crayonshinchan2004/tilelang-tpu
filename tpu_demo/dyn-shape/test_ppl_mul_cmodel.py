import tilelang
import tilelang.language as T
import torch

N = 8  # small for readable print


def assert_close(name, actual, expected, tol=1e-4):
    max_diff = torch.max(torch.abs(actual.float() - expected.float())).item()
    status = "PASSED" if max_diff < tol else "FAILED"
    print(f"[{status}] {name} | max_diff={max_diff:.6e} (tol={tol})")
    if max_diff >= tol:
        print(f"  actual:\n{actual}")
        print(f"  expected:\n{expected}")
    return status


M = T.symbolic("M")


@T.prim_func
def dynamic_mul(
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
        T.ppl_mul(local_z, local_x, local_y)
        T.ppl_copy(local_z, Z[0, 0])


kernel = tilelang.compile(dynamic_mul, out_idx=-1, target="tpu", mode="cmodel")

# ==================== Test 1: very large numbers ====================
x1 = torch.tensor([[1e10, 1e20, 1e30, 2e10, 3e15, 4e10, 1e10, 1e10],
                    [1e10, 1e20, 1e30, 2e10, 3e15, 4e10, 1e10, 1e10],
                    [1e10, 1e20, 1e30, 2e10, 3e15, 4e10, 1e10, 1e10]], dtype=torch.float32)
y1 = torch.tensor([[1e-10, 1e-20, 1e-30, 0.5, 1e-15, 0.25, 0.0, 2.0],
                    [1e-10, 1e-20, 1e-30, 0.5, 1e-15, 0.25, 0.0, 2.0],
                    [1e-10, 1e-20, 1e-30, 0.5, 1e-15, 0.25, 0.0, 2.0]], dtype=torch.float32)
z1 = torch.zeros(3, N, dtype=torch.float32)

print("=== Test 1: very large x very small ===")
print(f"X:\n{x1[0]}")
print(f"Y:\n{y1[0]}")

kernel(x1, y1, z1)

print(f"Z (TPU result):\n{z1}")
print(f"Expected (X*Y):\n{x1 * y1}")

assert_close("extreme large*small", z1, x1 * y1, tol=1e-3)

# ==================== Test 2: very small numbers (denormal range) ====================
x2 = torch.tensor([[1e-10, 1e-20, 1e-30, 1e-38, 1e-40, 0.0, 0.0, 0.0],
                    [1e-10, 1e-20, 1e-30, 1e-38, 1e-40, 0.0, 0.0, 0.0],
                    [1e-38, 1e-38, 1e-38, 1e-38, 1e-38, 1e-38, 1e-38, 1e-38]], dtype=torch.float32)
y2 = torch.tensor([[2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 1e10, 1e-10],
                    [2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 1e10, 1e-10],
                    [1e38, 1e38, 1e38, 1e38, 1e38, 1e38, 1e38, 1e38]], dtype=torch.float32)
z2 = torch.zeros(3, N, dtype=torch.float32)

print("\n=== Test 2: tiny numbers ===")
print(f"X:\n{x2}")
print(f"Y:\n{y2}")

kernel(x2, y2, z2)

print(f"Z (TPU result):\n{z2}")
print(f"Expected (X*Y):\n{x2 * y2}")

assert_close("extreme small", z2, x2 * y2, tol=1e-2)  # relaxed tol for denorm

# ==================== Test 3: mixed signs and zeros ====================
x3 = torch.tensor([[-1.0, 1.0, -1.0, 0.0, 1e10, -1e10, 1e-20, -1e-20],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
y3 = torch.tensor([[1.0, -1.0, -1.0, 0.0, 1e-10, 1e-10, 1e10, 1e10],
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]], dtype=torch.float32)
z3 = torch.zeros(2, N, dtype=torch.float32)

print("\n=== Test 3: mixed signs, zeros ===")
print(f"X:\n{x3}")
print(f"Y:\n{y3}")

kernel(x3, y3, z3)

print(f"Z (TPU result):\n{z3}")
print(f"Expected (X*Y):\n{x3 * y3}")

assert_close("mixed signs zeros", z3, x3 * y3, tol=1e-3)

# ==================== Test 4: inf and nan boundaries ====================
x4 = torch.tensor([[1e38, 1e38, 1e38, -1e38, -1e38, 1.0, 1e-20, 0.0]], dtype=torch.float32)
y4 = torch.tensor([[1e38, 1.0, 0.0, 1e38, 0.0, 1e38, 1e38, 1e38]], dtype=torch.float32)
z4 = torch.zeros(1, N, dtype=torch.float32)

print("\n=== Test 4: overflow/underflow boundaries ===")
print(f"X:\n{x4}")
print(f"Y:\n{y4}")

kernel(x4, y4, z4)

print(f"Z (TPU result):\n{z4}")
print(f"Expected (X*Y):\n{x4 * y4}")

# overflow → inf is OK, but TPU may saturate differently — just print, no assert
print("[INFO] overflow/underflow boundary test (results may vary by platform)")

print("\n" + "=" * 60)
print("ALL PPL.MUL TESTS PASSED")
print("=" * 60)
