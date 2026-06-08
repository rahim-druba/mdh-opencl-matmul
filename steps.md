# Generate OpenCL Kernels for matmul

Open a terminal and paste these commands one by one:

```bash
cd "/home/rahim/pact_2019_artifact/preliminary/MLP&SVM/md_hom_generator/build"
```

```bash
cmake .. -Wno-dev
```

```bash
make matmul
```

```bash
./matmul
```

```bash
mv matmul_1.cl matmul_2.cl /home/rahim/mdh-output/
```

```bash
ls -lh /home/rahim/mdh-output/
```

**Output will be at:**
```
/home/rahim/mdh-output/matmul_1.cl
/home/rahim/mdh-output/matmul_2.cl
```
