
## Structure
All CUDA files are in the `CLAD` directory, which is organized as follows:


```
CLAD
├── include     -> All header files
│   └── ptx     -> Definitions of inline ptx fp/fr functions
├── src
│   ├── fp/             -> source files for functions on fp
│   ├── fr/             -> source files for functions on fr
│   └── matrixMult.cu   -> Extern functions for Loading sparse matrices and calculate witness multiplication.
└── tests
    ├── include/            -> Headers for test functions only
    ├── testFields/         -> Templated tests on fp and fr
    ├── testG1/             -> Tests for G1 curve
    ├── test_fp.cu          -> standalone tests for FP
    ├── test_fr.cu          -> standalone tests for FR
    ├── test_matrix_mult.cu -> Standalone test and simple bench for matrix multiplication.
    └── test_util.cu        -> common utilities
```

## Running

For running tests, in the clad directory, call

`make <testname>`

With `<testname>` being:
- `mmul` for matrix multiplication;
- `fp` for sacalars in Fp;
- `fr` for sacalars in Fr;

