use std::ops::{Index, IndexMut};

//for testing
extern "C" {
    fn scratchboard_cuda();
}

//for testint
pub fn scratchboard() {
    unsafe {
        scratchboard_cuda(); // Call the CUDA function
    }
}

//data types
// pub type FpT = [u64; 4];
// pub type FrT = [u64; 4];

#[repr(C)]
#[derive(Debug, Default)]
pub struct FpT {
    data: [u64; 4],
}
#[repr(C)]
#[derive(Debug, Default)]
pub struct FrT {
    data: [u64; 4],
}

#[repr(C)]
#[derive(Debug, Default)]
pub struct G1aT {
    pub x: FpT,
    pub y: FpT,
}

#[repr(C)]
#[derive(Debug, Default)]
pub struct G1pT {
    pub x: FpT,
    pub y: FpT,
    pub z: FpT
}

// entry point for the msm
extern "C"{
    fn msm(out: *mut G1aT, points: *const G1aT, scalars: *const FrT, nPoints: usize);
}


// pub fn cuda_msm(points: &[G1aT], scalars: &[FrT]) -> G1aT {
pub fn cuda_msm(points: &[G1aT], scalars: &[FrT]) -> G1aT {
    let n_points = points.len();
    assert!(n_points == scalars.len(), "lenght mismatch"); 

    let mut ret = G1aT::default();

    // clad_msm(&mut ret, &points[0], &scalars[0], npoints);

    unsafe { msm(&mut ret, &points[0], &scalars[0], n_points) };

    ret
}


//util for accessing the indexes of FP and FR
impl Index<usize> for FpT {
    type Output = u64;

    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < 4);
        &self.data[index]
    }
}

impl IndexMut<usize> for FpT {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        assert!(index < 4);
        &mut self.data[index]
    }
}

impl FpT {
    pub fn new(x: u64) -> Self {
        Self {
            data: [x, 0, 0, 0],
        }
    }

    pub fn from_array(array: [u64; 4]) -> Self {
        Self { data: array }
    }
}

//------------------------------------------------------------------------------

impl Index<usize> for FrT {
    type Output = u64;

    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < 4);
        &self.data[index]
    }
}

impl IndexMut<usize> for FrT {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        assert!(index < 4);
        &mut self.data[index]
    }
}

impl FrT {
    pub fn from_array(array: [u64; 4]) -> Self {
        Self { data: array }
    }
}
