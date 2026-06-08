# MDH OpenCL — Matrix Multiplication

> Automatic generation of portable, high-performance OpenCL GPU kernels for **Matrix Multiplication** using the **Multi-Dimensional Homomorphism (MDH)** framework.
> Based on the PACT 2019 paper by Rasch, Schulze & Gorlatch.

---

## Operation

$$S[L_1, L_2] = \sum_{R_1} Z[L_1, R_1] \times W[L_2, R_1]$$

Pure matrix multiplication — no bias, no activation. Every output element `S[i,j]` is the dot product of row `i` from `Z` and row `j` from `W`, summed over the reduction dimension `R1`.

| Symbol | Shape | Description |
|--------|-------|-------------|
| `Z` | `[L1, R1]` | Input matrix (batch × input features) |
| `W` | `[L1, L2, R1]` | Weight matrix (output features × input features) |
| `S` | `[L1, L2]` | Output matrix (batch × output features) |

---

## Repository Structure

```
mdh-opencl-matmul/
│
├── input/
│   └── matmul.cpp          # MDH high-level specification for matrix multiplication
│
├── generated/
│   ├── matmul_1.cl         # Generated OpenCL kernel 1 — tiled multiply phase (~20 MB)
│   └── matmul_2.cl         # Generated OpenCL kernel 2 — reduction phase (~14 MB)
│
├── implemented.md           # Complete step-by-step generation log
├── proof.md                 # Correctness proof via comparison with MLP kernel
└── steps.md                 # Commands to reproduce the generation yourself
```

---

## How the Kernels Are Generated

The file `input/matmul.cpp` is a **high-level MDH specification** — it describes the operation mathematically using the MDH C++ DSL:

```cpp
auto f = md_hom::scalar_function("return Z_val * W_val;");  // element-wise multiply
auto g = md_hom::scalar_function("return res;");            // identity (no activation)

auto md_hom_matmul = md_hom::md_hom<2, 1>(
    "matmul",
    md_hom::inputs(Z, W),
    f, g, result,
    true, true
);
```

This is compiled and run through the **MDH OpenCL generator**, which produces two kernel files (`matmul_1.cl`, `matmul_2.cl`) encoding **every possible tiling, work-group size, and memory hierarchy configuration** as preprocessor branches.

At OpenCL compile time, the optimal variant is selected by passing `-D` flags:

```bash
-D OCL_DIM_L_1=1  -D OCL_DIM_L_2=0  -D OCL_DIM_R_1=2  \
-D G_CB_SIZE_L_1=64  -D G_CB_SIZE_L_2=64  -D G_CB_SIZE_R_1=32
```

This makes the kernels **portable across GPUs** — the same source adapts to any device through auto-tuning.

---

## Reproducing the Generation

All commands needed to regenerate `matmul_1.cl` and `matmul_2.cl` from scratch are in [`steps.md`](steps.md).

For the full step-by-step process including build logs, see [`implemented.md`](implemented.md).  
For a correctness comparison against the MLP kernel, see [`proof.md`](proof.md).

---

## Related Work

| Repo | Description |
|------|-------------|
| [mdh-opencl-generator-test](https://github.com/rahim-druba/mdh-opencl-generator-test) | Previous work — MLP forward pass `S = σ(Z × W + B)` generated using the same framework |
| [pact_2019_artifact](https://gitlab.com/mdh-project/pact_2019_artifact) | Original upstream MDH generator and full benchmark suite |

---

## Citation

```bibtex
@inproceedings{rasch2019mdh,
  title     = {Generating Portable High-Performance Code via Multi-Dimensional Homomorphisms},
  author    = {Rasch, Ari and Schulze, Richard and Gorlatch, Sergei},
  booktitle = {Proceedings of the 28th International Conference on
               Parallel Architectures and Compilation Techniques (PACT)},
  year      = {2019}
}
```
