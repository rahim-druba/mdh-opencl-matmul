# MDH OpenCL Matmul

Generating optimized OpenCL GPU kernels for **Matrix Multiplication** using the Multi-Dimensional Homomorphism (MDH) framework from the PACT 2019 paper.

## Operation

$$S[L_1, L_2] = \sum_{R_1} Z[L_1, R_1] \times W[L_2, R_1]$$

Pure matrix multiplication - no bias, no activation. Every output element `S[i,j]` is the dot product of row `i` from `Z` and row `j` from `W`, summed over the reduction dimension `R1`.

| Symbol | Shape | Description |
|--------|-------|-------------|
| `Z` | `[L1, R1]` | Input matrix (batch x input features) |
| `W` | `[L1, L2, R1]` | Weight matrix (output features x input features) |
| `S` | `[L1, L2]` | Output matrix (batch x output features) |

## Related

- **MLP version (previous work):** https://github.com/rahim-druba/mdh-opencl-code-generation-test
  - The MLP forward pass (`S = σ(Z × W + B)`) was generated there. This repo extends that work to plain matrix multiplication.
- **Original upstream generator:** https://gitlab.com/mdh-project/pact_2019_artifact
  - PACT 2019 artifact by Rasch, Schulze, Gorlatch.

## Files

```
input/matmul.cpp          # MDH specification for matrix multiplication
generated/matmul_1.cl     # Generated OpenCL kernel 1 - tiled multiply phase
generated/matmul_2.cl     # Generated OpenCL kernel 2 - reduction phase
implemented.md            # Full step-by-step log of how this was generated
proof.md                  # Comparison with MLP kernel proving correctness
steps.md                  # Commands to re-generate the kernels yourself
```

## How It Was Generated

The file `input/matmul.cpp` is a **high-level MDH specification** - it describes the operation mathematically using the MDH C++ DSL:

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

See `implemented.md` for the full step-by-step process and `steps.md` to reproduce it.

## Citation

```
@inproceedings{rasch2019mdh,
  title={Generating Portable High-Performance Code via Multi-Dimensional Homomorphisms},
  author={Rasch, Ari and Schulze, Richard and Gorlatch, Sergei},
  booktitle={PACT},
  year={2019}
}
```
