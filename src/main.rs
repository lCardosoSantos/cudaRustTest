extern crate cuda_test;

use cuda_test::msm;


// fn main(){
//     println!("Leaving rust...");
//     msm::scratchboard();
// }


// //actual main function
fn main(){
    println!("MSM:");

    let mut out = msm::G1aT::default();

    let mut points: [msm::G1aT; 4] = Default::default();
    for i in 0..4 {
        points[i] = msm::G1aT::default();
    }

    let mut scalars: [msm::FrT; 4] = Default::default();
    for i in 0..4 {
        scalars[i] = msm::FrT::from_array([i as u64, 0 as u64, 0 as u64, 0 as u64]);
    }

    println!("Points: {:?}", points);
    println!("scalars: {:?}", scalars);

    out = msm::cuda_msm(&points, &scalars);

    println!("out:  {:?}", out);
}
