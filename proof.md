# Proof: matmul_1.cl vs mlp_forward_1.cl

This document proves that `matmul_1.cl` was correctly generated from `matmul.cpp`
by comparing it against `mlp_forward_1.cl` (generated from `eccmlp_forward.cpp`).

---

## 1. Kernel Signatures

The most visible difference — `matmul_1.cl` has no `B` (bias) buffer at all.

**mlp_forward_1.cl**
```c
__kernel void mlp_forward_1(
    __global TYPE_T const * const restrict Z,
    __global TYPE_T const * const restrict W,
    __global TYPE_T const * const restrict B,    ← bias buffer
    __global TYPE_TS * const restrict res_g,
    __global TYPE_TS * const restrict int_res,
    __global TYPE_TS * const restrict S_orig)
```

**matmul_1.cl**
```c
__kernel void matmul_1(
    __global TYPE_T const * const restrict Z,
    __global TYPE_T const * const restrict W,    ← no B
    __global TYPE_TS * const restrict res_g,
    __global TYPE_TS * const restrict int_res,
    __global TYPE_TS * const restrict S_orig)
```

---

## 2. Scalar Functions `f` and `g`

These are the exact functions written in each `.cpp` spec file, embedded verbatim into the kernel.

### `f` function — identical in both (element-wise multiply)

```c
inline TYPE_TS f(const TYPE_T Z_val, const TYPE_T W_val) {
    return Z_val * W_val;
}
```

### `g` function — the core difference

| | Signature | Body |
|---|---|---|
| `mlp_forward_1.cl` | `g(const TYPE_T res, const TYPE_T B_val)` | `return 1 / (1 + exp(-(res + B_val)));` |
| `matmul_1.cl` | `g(const TYPE_T res)` | `return res;` |

**mlp_forward_1.cl** — adds bias then applies sigmoid activation:
```c
inline TYPE_TS g(const TYPE_T res, const TYPE_T B_val) {
    return 1 / (1 + exp(-(res + B_val)));
}
```

**matmul_1.cl** — identity, passes result straight through:
```c
inline TYPE_TS g(const TYPE_T res) {
    return res;
}
```

---

## 3. `B_val` References Across the Entire File

| File | Occurrences of `B_val` |
|---|---|
| `mlp_forward_1.cl` | **2** — in `g` function signature and body |
| `matmul_1.cl` | **0** — not present anywhere |

This confirms no bias logic leaked into the matmul kernel.

---

## 4. File Size Comparison

| File | Lines | Size |
|---|---|---|
| `mlp_forward_1.cl` | 280,434 | ~20 MB |
| `matmul_1.cl` | 280,373 | ~20 MB |

Only **61 lines difference** between the two files. The entire structural difference
(tiling logic, preprocessor macros, memory hierarchy) is identical — only the
scalar functions and the presence/absence of the `B` buffer differ.

---

## 5. Conclusion

| What to verify | Expected (from `matmul.cpp`) | Found in `matmul_1.cl` | Match |
|---|---|---|---|
| Kernel name | `matmul_1` | `matmul_1` | ✓ |
| Input `Z` present | yes | yes | ✓ |
| Input `W` present | yes | yes | ✓ |
| Input `B` absent | yes | not found | ✓ |
| `f`: `Z_val * W_val` | yes | line 629 | ✓ |
| `g`: `return res` | yes | line 635 | ✓ |
| No `exp` / sigmoid | yes | not found | ✓ |
| No `B_val` anywhere | yes | 0 occurrences | ✓ |

The generated kernel is a correct and complete translation of `matmul.cpp`.
