# MDH OpenCL Matmul

Generating optimized OpenCL GPU kernels for **Matrix Multiplication** using the Multi-Dimensional Homomorphism (MDH) framework from the PACT 2019 paper.

## Operation

```
S[L1, L2] = sum_R1( Z[L1, R1] × W[L2, R1] )
```

Pure matrix multiplication - no bias, no activation function.

## Related

- **MLP version (previous work):** https://github.com/rahim-druba/mdh-opencl-generator-test
  — The MLP forward pass (`S = σ(Z × W + B)`) was generated there. This repo extends that work to plain matrix multiplication.
- **Original upstream generator:** https://gitlab.com/mdh-project/pact_2019_artifact
  — PACT 2019 artifact by Rasch, Schulze, Gorlatch.

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

The `matmul.cpp` spec file was fed into the MDH OpenCL generator from the upstream repo.
The generator produced two kernel files encoding every possible tiling and work-group
configuration as preprocessor branches - selected at OpenCL compile time via `-D` flags
for optimal performance on any GPU.

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
