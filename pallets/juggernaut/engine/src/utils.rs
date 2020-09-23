use crate::{
    sample::Sample,
    matrix::Matrix,
    matrix::MatrixTrait,
};
pub fn sample_input_to_matrix(sample: &Sample) -> Matrix {
    let mut f64_vec: Vec<Vec<f64>> = vec![];

    f64_vec.push(sample.inputs.clone());

    return Matrix::generate(1, sample.get_inputs_count(), |m, n| f64_vec[m][n]);
}

pub fn sample_output_to_matrix(sample: &Sample) -> Matrix {
    let mut f64_vec: Vec<Vec<f64>> = vec![];

    f64_vec.push(sample.outputs.clone().unwrap());

    return Matrix::generate(1, sample.get_outputs_count(), |m, n| f64_vec[m][n]);
}
