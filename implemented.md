# Implementation Log: MDH OpenCL Kernel Generation for Matrix Multiplication

**Date:** 2026-06-08  
**Goal:** Generate OpenCL GPU kernels for matrix multiplication (`S = Z × W`) using the Multi-Dimensional Homomorphism (MDH) generator from the PACT 2019 research artifact.

---

## Overview

The MDH generator takes a high-level C++ specification file describing a mathematical operation and produces two optimized, auto-tunable OpenCL kernel files. This log documents every step taken to generate `matmul_1.cl` and `matmul_2.cl` from scratch.

---

## Step 1 — Clone the Upstream Repository

**What:** Cloned the full PACT 2019 artifact repository which contains the MDH generator library.

**Source:** `https://gitlab.com/mdh-project/pact_2019_artifact`

**Command:**
```bash
git clone https://gitlab.com/mdh-project/pact_2019_artifact /home/rahim/pact_2019_artifact
```

**Result:** Repository cloned to `/home/rahim/pact_2019_artifact/`

**Key folder used from this repo:**
```
/home/rahim/pact_2019_artifact/preliminary/MLP&SVM/md_hom_generator/
├── md_hom_generator.hpp   ← top-level include for all spec files
├── CMakeLists.txt         ← build configuration
├── include/               ← all generator C++ headers
└── src/                   ← generator implementation + existing spec files
    ├── eccmlp/            ← existing MLP spec files
    ├── svm/               ← existing SVM spec files
    ├── helper.cpp
    ├── input_buffer.cpp
    ├── input_scalar.cpp
    ├── md_hom.cpp
    ├── result_buffer.cpp
    └── scalar_function.cpp
```

---

## Step 2 — Create the matmul Spec File

**What:** Created a new directory `src/matmul/` inside the generator and wrote the MDH specification for pure matrix multiplication.

**Commands:**
```bash
mkdir -p "/home/rahim/pact_2019_artifact/preliminary/MLP&SVM/md_hom_generator/src/matmul"
```

**File created:**
```
/home/rahim/pact_2019_artifact/preliminary/MLP&SVM/md_hom_generator/src/matmul/matmul.cpp
```

**File content:**
```cpp
#include "../../md_hom_generator.hpp"

/**
 * L1: Batch size (number of samples)
 * L2: Number of output features (columns of W)
 * R1: Number of input features (columns of Z / rows of W)
 *
 * Computes: S[L1, L2] = sum_R1( Z[L1, R1] * W[L2, R1] )
 */
int main() {
    auto Z = md_hom::input_buffer("Z", {md_hom::L(1), md_hom::R(1)});
    auto W = md_hom::input_buffer("W", {md_hom::L(1), md_hom::L(2), md_hom::R(1)});
    auto result = md_hom::result_buffer("S", {md_hom::L(1), md_hom::L(2)});

    auto f = md_hom::scalar_function("return Z_val * W_val;");
    auto g = md_hom::scalar_function("return res;");  // identity — no activation

    auto md_hom_matmul = md_hom::md_hom<2, 1>(
        "matmul",
        md_hom::inputs(Z, W),
        f, g,
        result,
        true, true
    );

    auto generator = md_hom::generator::ocl_generator(md_hom_matmul);
    std::ofstream kernel_file;

    kernel_file.open("matmul_1.cl", std::fstream::out | std::fstream::trunc);
    kernel_file << generator.kernel_1();
    kernel_file.close();

    kernel_file.open("matmul_2.cl", std::fstream::out | std::fstream::trunc);
    kernel_file << generator.kernel_2();
    kernel_file.close();
}
```

**Key design decisions vs. the MLP example:**
| | MLP (`eccmlp_forward.cpp`) | matmul (`matmul.cpp`) |
|---|---|---|
| Inputs | Z, W, B (bias) | Z, W only |
| `f` function | `Z_val * W_val` | `Z_val * W_val` (same) |
| `g` function | `1/(1+exp(-(res+B_val)))` (sigmoid) | `return res` (identity) |
| Output | `S = σ(Z×W + B)` | `S = Z × W` |
| MDH template | `md_hom<2,1>` | `md_hom<2,1>` (same) |

---

## Step 3 — Add matmul Target to CMakeLists.txt

**What:** Edited the generator's `CMakeLists.txt` to register `matmul` as a new build target, following the exact same pattern used for all existing targets.

**File edited:**
```
/home/rahim/pact_2019_artifact/preliminary/MLP&SVM/md_hom_generator/CMakeLists.txt
```

**Lines added at the bottom:**
```cmake
# matmul
add_executable(matmul src/matmul/matmul.cpp)
target_link_libraries(matmul md_hom_generator)
```

**How it fits the existing pattern** — the full CMakeLists.txt structure is:
```cmake
cmake_minimum_required(VERSION 2.8.11)
project(md_hom_generator)
set(CMAKE_CXX_STANDARD 14)
include_directories(include)
file(GLOB_RECURSE SOURCE_FILES src/*.cpp)
add_library(md_hom_generator STATIC ${SOURCE_FILES})
...
# ECC+MLP (existing)
add_executable(eccmlp_forward src/eccmlp/eccmlp_forward.cpp)
target_link_libraries(eccmlp_forward md_hom_generator)
...
# matmul (added)
add_executable(matmul src/matmul/matmul.cpp)
target_link_libraries(matmul md_hom_generator)
```

---

## Step 4 — Build with CMake

**What:** Created a `build/` directory, ran CMake to configure, then compiled only the `matmul` target.

**Commands:**
```bash
mkdir -p "/home/rahim/pact_2019_artifact/preliminary/MLP&SVM/md_hom_generator/build"
cd "/home/rahim/pact_2019_artifact/preliminary/MLP&SVM/md_hom_generator/build"
cmake ..
make matmul
```

**CMake output (summary):**
```
-- The CXX compiler identification is GNU 9.4.0
-- Configuring done
-- Generating done
-- Build files have been written to: .../md_hom_generator/build
```

**Result:** Executable produced at:
```
/home/rahim/pact_2019_artifact/preliminary/MLP&SVM/md_hom_generator/build/matmul
```

**Note:** Compiler warnings during build are harmless — they come from the existing generator library source files and appear for all targets, not just matmul.

---

## Step 5 — Run the Generator and Move Output Files

**What:** Executed the `matmul` binary inside `build/`, which ran the MDH generator and wrote two OpenCL kernel files. Then moved them into the project's `generated/` directory.

**Commands:**
```bash
cd "/home/rahim/pact_2019_artifact/preliminary/MLP&SVM/md_hom_generator/build"
./matmul

mv matmul_1.cl matmul_2.cl /home/rahim/mdh-opencl-generator-test/generated/
```

**Files generated and their final locations:**

| File | Size | Location | Role |
|------|------|----------|------|
| `matmul_1.cl` | 20 MB | `/home/rahim/mdh-opencl-generator-test/generated/matmul_1.cl` | Kernel 1 — tiled Z×W multiply, writes partial sums to `int_res` |
| `matmul_2.cl` | 14 MB | `/home/rahim/mdh-opencl-generator-test/generated/matmul_2.cl` | Kernel 2 — reduction phase, writes final output `S` |

**Generated kernel signatures:**
```c
// matmul_1.cl
__kernel void matmul_1(__global TYPE_T *Z, __global TYPE_T *W,
                       __global TYPE_TS *res_g, __global TYPE_TS *int_res,
                       __global TYPE_TS *S_orig)

// matmul_2.cl
__kernel void matmul_2(__global TYPE_T *int_res,
                       __global TYPE_TS *res_g, __global TYPE_TS *S,
                       __global TYPE_TS *S_orig)
```

---

## Final State of the Project

```
/home/rahim/mdh-opencl-generator-test/
├── README.md
├── plan.md                  ← step-by-step plan (pre-execution)
├── implemented.md           ← this file
├── repo_analysis.md         ← analysis of local + upstream repos
├── .gitattributes
│
├── input/
│   └── eccmlp_forward.cpp   ← original MLP spec (from upstream repo)
│
└── generated/
    ├── matmul_1.cl          ← NEW: matmul kernel 1 (20 MB)
    ├── matmul_2.cl          ← NEW: matmul kernel 2 (14 MB)
    ├── mlp_forward_1.cl     ← existing: MLP kernel 1 (20 MB)
    └── mlp_forward_2.cl     ← existing: MLP kernel 2 (14 MB)
```

**Upstream repo location (generator source):**
```
/home/rahim/pact_2019_artifact/
└── preliminary/MLP&SVM/md_hom_generator/
    ├── src/matmul/matmul.cpp    ← spec file written in Step 2
    ├── build/matmul             ← compiled executable from Step 4
    └── CMakeLists.txt           ← edited in Step 3
```

---

## How the Generated Kernels Work

The `.cl` files are not ordinary single-configuration kernels. They encode **every possible tile size, work-group dimension, and memory level assignment** as `#if`/`#elif` preprocessor branches. At OpenCL runtime, you select the optimal configuration by passing compile-time macros:

```bash
-D OCL_DIM_L_1=1 -D OCL_DIM_L_2=0 -D OCL_DIM_R_1=2 \
-D G_CB_SIZE_L_1=64 -D G_CB_SIZE_L_2=64 -D G_CB_SIZE_R_1=32 \
-D L_CB_SIZE_L_1=8  -D L_CB_SIZE_L_2=8  -D L_CB_SIZE_R_1=8  \
-D P_CB_SIZE_L_1=1  -D P_CB_SIZE_L_2=1  -D P_CB_SIZE_R_1=1
```

This is what makes the kernels **portable and high-performance** across different GPUs — the same source file adapts to any hardware via auto-tuning (finding the best macro values for your specific device).