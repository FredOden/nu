use rand::Rng;

use std::f64::consts::E;

pub struct Neuron {
    weights: Vec<f64>,
    bias: f64,
}

impl Neuron {
    // Creates a new neuron with random weights and bias
    pub fn new(input_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights = (0..input_size).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let bias = rng.gen_range(-1.0..1.0);

        Neuron { weights, bias }
    }

    // The sigmoid activation function
    pub fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + E.powf(-x))
    }

    // Performs the forward pass with sigmoid activation
    pub fn forward(&self, inputs: &[f64]) -> f64 {
        assert_eq!(inputs.len(), self.weights.len());
        let weighted_sum: f64 = inputs.iter().zip(self.weights.iter()).map(|(&input, &weight)| input * weight).sum();
        Neuron::sigmoid(weighted_sum + self.bias)
    }
}

// Example usage:
// let neuron = Neuron::new(3); // for 3 input connections
// let inputs = vec![0.5, -0.1, 0.8];
// let output = neuron.forward(&inputs);

