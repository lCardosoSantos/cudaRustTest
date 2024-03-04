
<!--Temporary readme, will be better organized when the layout of the project is better defined
For now, it keeps track of important info about the code.
TODO: Update main readme
-->

## Structure

├── include -> All header files
├── src     -> MSM functions
│   ├── fp      -> Functions to manipulate the curve field
│   ├── fr      -> Functions to manipulate the scalar field
│   └── g1      -> Functions to manipulate the elliptic curve
└── tests   -> Outside Testing platforms, can be exposed to Rust; Main tests.
    ├── fpTest
    ├── frTest
    ├── g1Test
    ├── include -> All header files
    ├── msmTest
    ├── fp_test.cu
    ├── fr_test.cu
    ├── g1_test.cu
    └── msm_test.cu

Lower level functions (fp, fr, g1) should not be exposed to the rust code.

