
<!--Temporary readme, will be better organized when the layout of the project is better defined
For now, it keeps track of important info about the code.
TODO: Update main readme
-->

## Structure

```
Lower level functions (fp, fr, g1) should not be exposed to the rust code.
.
├── include -> All header files
│   └── ptx     -> PTX related headers
│       ├── fp/      -> Inline definition of functions on fp (overloaded with fr)
│       └── fr/      -> Inline definition of functions on fr (overloaded with fp)
├── src
│   ├── fp/      -> cuda files for functions on fp (overloaded with fr)
│   ├── fr/      -> cuda files for functions on fp (overloaded with fr)
│   └── g1/      -> cuda files for functions on g1
│   └── msm.cu   -> Functions exposed to RUST
└── tests
    ├── include/       -> Headers for test functions only
    ├── testFields/    -> templated tests on fp and fr
    ├── testG1/         -> tests on g1
    ├── test_fp.cu     -> standalone tests
    ├── test_fr.cu     -> standalone tests
    ├── test_g1.cu     -> standalone tests
    ├── test_msm.cu    -> standalone tests
    └── test_util.cu   -> common utilities
```

## todo:
- [ ] Makefile build (note 1)
- [ ] Rust build
  - [ ] lib
  - [ ] test
- [ ] add dummy code to function definitions and test for full integration


Note 1: Making the makefile a bit smarter would be usefull for possible extensions. I would say that implementing C++ namespaces would make it simpler
Note 2: Does this works?

```C
template<typename T>
__global__ void kernel(){...}

template __global__ void kernel<type_t>();

int main(void) {
    kernel<type_t><<<block,grid>>>();
    SYNC;
    return 0;
}
```
On the same ideia, if I have a function to encapsulate kernels for test/benchmark, could I explicitly instantiate a templated kernel, and pass that as a function pointer?
