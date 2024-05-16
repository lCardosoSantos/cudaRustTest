
## Structure

```
Lower level functions (fp, fr, g1) should not be exposed to the rust code.
.
├── include -> All header files
│   └── ptx     -> PTX related headers
│       ├── fp/      -> Inline definition of functions on fp (overloaded with fr)
│       └── fr/      -> Inline definition of functions on fr (overloaded with fp)
├── src
│   ├── fp/             -> cuda files for functions on fp (overloaded with fr)
│   ├── fr/             -> cuda files for functions on fp (overloaded with fr)
│   └── g1/             -> cuda files for functions on g1 (deprecated)
│   └── matrixMult.cu   -> Functions exposed to RUST
└── tests
    ├── include/       -> Headers for test functions only
    ├── testFields/    -> templated tests on fp and fr
    ├── testG1/        -> tests on g1 (deprecated)
    ├── test_fp.cu     -> standalone tests
    ├── test_fr.cu     -> standalone tests
    ├── test_g1.cu     -> standalone tests (deprecated)
    ├── test_msm.cu    -> standalone tests (derpecated)
    ├── test_matrix_mult.cu -> Standalone test and bench for matrix multiplication. See note.
    └── test_util.cu   -> common utilities
```
