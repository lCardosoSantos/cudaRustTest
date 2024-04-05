// extern crate grumpkin_msm;
// use grumpkin_msm::call_clad;

extern crate cuda_test;


mod tests{

    #[cfg(feature = "msm_primitive_test")]
    extern "C"{
        fn run_fr_tests();
    }

    #[cfg(feature = "msm_primitive_test")]
    extern "C"{
        fn run_fp_tests();
    }

    // extern "C"{
    //     fn run_msm_tests();
    // }


    #[cfg(feature = "msm_primitive_test")]
    pub fn call_run_fr_tests(){
        unsafe {run_fr_tests()};
    }

    #[cfg(feature = "msm_primitive_test")]
    pub fn call_run_fp_tests(){
        unsafe {run_fp_tests()};
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

    #[cfg(feature = "msm_primitive_test")]
    #[test]
    fn test_fields(){
               
        call_run_fp_tests();
        call_run_fr_tests();
        assert!(true);
    }
    
    // #[cfg(feature = "msm_primitive_test")]
    // #[test]
    // fn test_elliptic_curve(){

    //     call_run_g1_tests();
    //     assert!(false);
    // }

}
