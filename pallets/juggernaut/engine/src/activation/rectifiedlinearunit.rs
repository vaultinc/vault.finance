use crate::activation::Activation;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize,Copy,Clone)]
pub struct RectifiedLinearUnit;

impl RectifiedLinearUnit {
    pub fn new() -> RectifiedLinearUnit {
        return RectifiedLinearUnit;
    }
}

#[typetag::serde(name= "RectifiedLinearUnit")]
impl Activation for RectifiedLinearUnit {
    /// Calculates the RectifiedLinearUnit of input `x`
    fn calc(&self, x: Vec<f64>) -> Vec<f64> {
        x.iter()
            .map(|&n| if n <= 0f64 { 0f64 } else { n })
            .collect::<Vec<_>>()
    }

    /// Calculates the Derivative RectifiedLinearUnit of input `x`
    fn derivative(&self, x: Vec<f64>) -> Vec<f64> {
        x.iter()
            .map(|&n| if n <= 0f64 { 0f64 } else { n })
            .collect::<Vec<_>>()
    }
}

#[cfg(test)]
mod tests {
    use super::Activation;
    use super::RectifiedLinearUnit;
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn rectifiedlinearunit_test() {
        let activation = RectifiedLinearUnit::new();
        assert_approx_eq!(activation.calc(vec![3.4f64])[0], 3.4f64);
        assert_approx_eq!(activation.calc(vec![-3.4f64])[0], 0f64);
    }

    #[test]
    fn rectifiedlinearunit_derivative_test() {
        let activation = RectifiedLinearUnit::new();
        assert_approx_eq!(activation.derivative(vec![-3.4f64])[0], 0f64);
        assert_approx_eq!(activation.derivative(vec![3.4f64])[0], 3.4f64);
    }
}
