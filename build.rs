use std::path::Path;

fn main() {
    // // Define the CUDA source files directory
    let clad_dir = Path::new("src/CLAD/src");
    let clad_include_dir = Path::new("src/CLAD/include");
    let clad_include_ptx_dir = Path::new("src/CLAD/include/ptx");

    
    // Find all CUDA files recursively within the CLAD directory
    let mut cuda_files = find_cuda_files(clad_dir);
    
    if cfg!(feature="msm_primitive_test") {
        cuda_files.append(&mut find_cuda_files(Path::new("src/CLAD/tests"))); //TODO: Update to run tests without the feature(basic tests only)
    }
    // Compile each CUDA file individually
    println!("cargo:warning= Source files: {:?}", cuda_files);
    
    let mut nvcc = cc::Build::new();
    nvcc.cuda(true);
    nvcc.cargo_debug(true);
    // nvcc.cudart("static");
    nvcc.include(clad_include_dir);
    nvcc.include(clad_include_ptx_dir);

    if cfg!(feature="msm_primitive_test"){ 
        let clad_test_include_dir = Path::new("src/CLAD/tests/include");
        nvcc.include(clad_test_include_dir);
        nvcc.define("RUST_TEST", None);
    }

    nvcc.no_default_flags(true);
    nvcc.warnings(false);
    
    nvcc.clone().files(cuda_files). 
    compile("clad");

    println!("cargo:rerun-if-changed=src/CLAD");
    
}

// Function to find CUDA files recursively within a directory
fn find_cuda_files(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut cuda_files = Vec::new();
    for entry in dir.read_dir().expect("Failed to read directory") {
        let entry = entry.expect("Failed to get directory entry");
        let path = entry.path();
        if path.is_file() {
            if let Some(ext) = path.extension() {
                if ext == "cu" {
                    cuda_files.push(path);
                }
            }
        } else if path.is_dir() {
            if !path.starts_with("test"){ //bodge for testing
                cuda_files.extend(find_cuda_files(&path));
            }
        }
    }
    cuda_files
}
