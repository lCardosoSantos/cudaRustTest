// extern crate grumpkin_msm;
// use grumpkin_msm::call_clad;

extern crate cuda_test;

mod tests{
    #[test]
    fn base_test(){
        println!("base test");
        cuda_test::msm::scratchboard();
        assert!(false);
    }

    #[cfg(feature = "msm_primitive_test")]
    #[test]
    fn testFields(){
        println!("testFileds");
        assert!(false);
    }
    
    #[cfg(feature = "msm_primitive_test")]
    #[test]
    fn testEllipticCurve(){
        println!("testEllipticCurve");
        assert!(false);
    }

}
