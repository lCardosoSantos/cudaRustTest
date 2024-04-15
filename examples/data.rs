use std::fs::File;
use std::io::BufReader;

use bincode;
use home;
use pasta_curves::pallas;
use serde::de::DeserializeOwned;
use serde::Deserialize;

use crate::sparse::SparseMatrix;

mod sparse;

/// Path to the directory where Arecibo data will be stored.
pub static ARECIBO_DATA: &str = ".arecibo_data";

/// Reads and deserializes data from a specified section and label.
pub fn read_arecibo_data<T: DeserializeOwned>(
  section: &String,
  label: &String,
) -> T {
    let root_dir = home::home_dir().unwrap().join(ARECIBO_DATA);
    let section_path = root_dir.join(section);
    assert!(section_path.exists(), "Section directory does not exist");

    let file_path = section_path.join(label);
    assert!(file_path.exists(), "Data file does not exist");

    let file = File::open(file_path).expect("Failed to open data file");
    let reader = BufReader::new(file);

    bincode::deserialize_from(reader).expect("Failed to read data")
}

fn test_witness(i: i32){
    // the section and label values are hard coded,
    // you should inspect the generated files to see what your values are
    let section = "witness_0x01913e1d49eb0dc298c71bfcc8f75d2a1963276c3888f9885816ba948b480a2c";
    let label_i = format!("len_7999845_{}", i);
    let witness_i: Vec<pallas::Scalar> = read_arecibo_data(&section.into(), &label_i);
    println!("{}/{}", section, label_i);

    println!("Size of witness: {} Scalars", witness_i.len());
    for i in 0..7 {
        println!("\tElement {}: {:?}", i, witness_i[i]);
    }
    println!("...");
}

fn test_crossterm(i: i32){
    // the section and label values are hard coded,
    // you should inspect the generated files to see what your values are
    let section = "cross_term_0x01913e1d49eb0dc298c71bfcc8f75d2a1963276c3888f9885816ba948b480a2c";
    let label_i = format!("len_9825045_{}", i);
    let crossterm_i: Vec<pallas::Scalar> = read_arecibo_data(&section.into(), &label_i);
    println!("{}/{}", section, label_i);

    println!("Size of Crossterm: {} Scalars", crossterm_i.len());
    for i in 0..7 {
        println!("\tElement {}: {:?}", i, crossterm_i[i]);
    }
    println!("...");
}

fn test_matrix(label: &String){
    // the section and label values are hard coded,
    // you should inspect the generated files to see what your values are
    let section = "sparse_matrices_0x01913e1d49eb0dc298c71bfcc8f75d2a1963276c3888f9885816ba948b480a2c";
    let matrix: SparseMatrix<pallas::Scalar> = read_arecibo_data(&section.into(), &label);

    println!("SparseMatrix {}:", label);
    println!("data len = {}", matrix.data.len());
    println!("indices len = {}", matrix.indices.len());
    println!("indices max = {}", matrix.indices.iter().max().unwrap());
    println!("indpt len = {}", matrix.indptr.len());
    println!("cols = {}", matrix.cols);
    println!("Dimensions = {} rows x {} col", matrix.indptr.len()-1,
                                              matrix.indices.iter().max().unwrap_or(&0));
}

fn main() {
    test_witness(3);
    test_crossterm(3);
    test_matrix(&"A_0".to_string());
    test_matrix(&"B_1".to_string());
    test_matrix(&"C_2".to_string());
}
