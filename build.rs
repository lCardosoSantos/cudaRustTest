use std::path::Path;

fn main() {
    // // Define the CUDA source files directory
    // let clad_dir = Path::new("CLAD");
    
    // // Find all CUDA files recursively within the CLAD directory
    // let cuda_files = find_cuda_files(clad_dir);

    // // Compile each CUDA file individually
    // println!("cargo:warning={:?}", cuda_files);
    
    let mut nvcc = cc::Build::new();
    nvcc.cuda(true);
    nvcc.cudart("static");
    nvcc.clone().file("src/test.cu").compile("clad");

    
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
            cuda_files.extend(find_cuda_files(&path));
        }
    }
    cuda_files
}
