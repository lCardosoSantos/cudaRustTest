// Copyright 2022-2024 Dag Arne Osvik
// Copyright 2022-2024 Luan Cardoso dos Santos

// extern crate grumpkin_msm;
// use grumpkin_msm::call_clad;

extern crate cuda_test;

mod tests{
    use std::ffi::c_void;
    use crate::cuda_test::sparse;

    #[cfg(feature = "primitive_test")]
    extern "C"{
        fn run_fr_tests();
    }

    #[cfg(feature = "primitive_test")]
    extern "C"{
        fn run_fp_tests();
    }

    // #[cfg(feature = "primitive_test")]
    // extern "C"{
    //     fn run_sparseMatrix_test();
    // }

    extern "C" {
        fn run_matrixMult_tests();
    }

    #[cfg(feature = "primitive_test")]
    pub fn call_run_fr_tests(){
        unsafe {run_fr_tests()};
    }

    #[cfg(feature = "primitive_test")]
    pub fn call_run_fp_tests(){
        unsafe {run_fp_tests()};
    }
    
    pub fn call_run_sparse_matrix_mul(){
        unsafe {run_matrixMult_tests()};
    }


    // pub fn call_run_MSM_tests(){
    //     println!("Calling CUDA run_msm_tests()");
    //     println!("--\n");
    //     unsafe {run_msm_tests()};
    // }

    #[test]
    fn base_test(){
        // cuda_test::msm::scratchboard();
        // call_run_MSM_tests();
        call_run_sparse_matrix_mul();
        assert!(true);
    }

    #[cfg(feature = "primitive_test")]
    #[test]
    fn test_fields(){
        // call_run_fp_tests();
        // call_run_fr_tests();
        assert!(true);
    }

}
