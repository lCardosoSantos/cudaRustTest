
## Structure
All CUDA files are in the `CLAD` directory, which is organized as follows:


```
CLAD
├── include     -> All header files.
│   └── ptx     -> Definitions of inline ptx fp/fr functions.
├── src
│   ├── fp/             -> Source files for functions on fp.
│   ├── fr/             -> Source files for functions on fr.
│   └── matrixMult.cu   -> Extern functions for loading sparse matrices and calculate witness multiplication.
└── tests
    ├── include/            -> Headers for test functions only.
    ├── testFields/         -> Templated tests on fp and fr.
    ├── testG1/             -> Tests for G1 curve.
    ├── test_fp.cu          -> Standalone tests for FP.
    ├── test_fr.cu          -> Standalone tests for FR.
    ├── test_matrix_mult.cu -> Standalone test and simple bench for SpMV.
    └── test_util.cu        -> Common utilities.
```

## Running

For running tests, in the CLAD directory, call:

`make <testname>`

With `<testname>` being:
- `SpMV` for SpMV;
- `fp` for sacalars in Fp;
- `fr` for sacalars in Fr;

