
#[link(name = "clad")]
extern "C" {
    fn clad();
}

pub fn call_clad() {
    unsafe {
        clad(); // Call the CUDA function
    }
}



fn main() {
    println!("Hello, world!");

    call_clad()
}
