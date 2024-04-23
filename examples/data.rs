use std::ffi::c_void;
use std::{
    fs::{self, File},
    io::BufReader,
    sync::Mutex,
    time::Instant,
};

use bincode;
use home;
use serde::de::DeserializeOwned;
use serde::Deserialize;
use once_cell::sync::OnceCell;
use camino::{Utf8Path, Utf8PathBuf};
use halo2curves::bn256;

extern crate cuda_test;
use cuda_test::sparse::SparseMatrix;

/// Path to the directory where Arecibo data will be stored.
pub static ARECIBO_DATA: &str = ".arecibo_data";

/// Global configuration for Arecibo data storage, including root directory and counters.
/// This configuration is initialized on first use.
pub static ARECIBO_CONFIG: OnceCell<Mutex<DataConfig>> = OnceCell::new();

/// Configuration for managing Arecibo data files, including the root directory,
/// witness counter, and cross-term counter for organizing files.
#[derive(Debug, Clone, Default)]
pub struct DataConfig {
    root_dir: Utf8PathBuf,
}

pub fn init_config() -> Mutex<DataConfig> {
    let root_dir = home::home_dir().unwrap().join(ARECIBO_DATA);
    let root_dir = Utf8PathBuf::from_path_buf(root_dir).unwrap();
    if !root_dir.exists() {
        fs::create_dir_all(&root_dir).expect("Failed to create arecibo data directory");
    }

    let config = DataConfig { root_dir };

    Mutex::new(config)
}

pub fn read_arecibo_data<T: DeserializeOwned>(
    section: impl AsRef<Utf8Path>,
    label: impl AsRef<Utf8Path>,
) -> T {
    let mutex = ARECIBO_CONFIG.get_or_init(init_config);
    let config = mutex.lock().unwrap();

    let section_path = config.root_dir.join(section.as_ref());
    assert!(
        section_path.exists(),
        "Section directory does not exist: {}",
        section_path
    );

    // Assuming the label uniquely identifies the file, and ignoring the counter for simplicity
    let file_path = section_path.join(label.as_ref());
    assert!(
        file_path.exists(),
        "Data file does not exist: {}",
        file_path
    );

    let file = File::open(file_path).expect("Failed to open data file");
    let reader = BufReader::new(file);

    bincode::deserialize_from(reader).expect("Failed to read data")
}

extern "C" {
    fn sparseMatrix_read_test_w(
        data: *mut c_void,
        indices: *mut usize,
        indptr: *mut usize,
        rows: usize,
        cols: usize,
        nElements: usize,
    );
}

extern "C" {
    fn easy(data: *const c_void);
}


fn main() {
    let hash = "0x00dcf8387ba48c3015a47ec14b485aff3edaf92a7cef06410bf164ff7b45bea6";
    // let mut w : Vec<bn256::Fr> = read_arecibo_data(format!("witness_{}", hash), format!("_{}", 1));
    let mut w:Vec<u64> = vec![0x1, 0x2, 0x3, 0x4];

    println!("this is a pointer: {:?}", w.as_ptr());
    println!("[0] {:?}", w[0]);
    unsafe {
        easy(w.as_ptr().cast());
    }
    //passing A to C++
    // unsafe {
    //     sparseMatrix_read_test_w(A.data.as_mut_ptr().cast(),
    //                            A.indices.as_mut_ptr(),
    //                            A.indptr.as_mut_ptr(),
    //                            A.indptr.len() - 1,
    //                            A.cols,
    //                            A.data.len()
    //     )
    // }
    //Test communication with CUDA

    /*
    how should be done:
    all static pointers on the input

    output will work like this: Return a pointer and a leng, and use them to create a Vec from the raw data
     */
}
