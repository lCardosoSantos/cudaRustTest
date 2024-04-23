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

    #[cfg(feature = "primitive_test")]
    extern "C"{
        fn run_sparseMatrix_test();
    }


    #[cfg(feature = "primitive_test")]
    pub fn call_run_fr_tests(){
        unsafe {run_fr_tests()};
    }

    #[cfg(feature = "primitive_test")]
    pub fn call_run_fp_tests(){
        unsafe {run_fp_tests()};
    }
    
    #[cfg(feature = "primitive_test")]
    pub fn call_run_sparse_matrix_tests(){
        unsafe {run_sparseMatrix_test()};
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
        assert!(true);
    }

    #[cfg(feature = "primitive_test")]
    #[test]
    fn test_fields(){
        call_run_fp_tests();
        call_run_fr_tests();
        assert!(true);
    }

    #[cfg(feature = "primitive_test")]
    #[test]
    fn test_sparse_matrix(){
        call_run_sparse_matrix_tests();
        call_sparseMatrix_read_test();
        assert!(true);
    }

    //Todo: test data transfer to Cuda side.

}
