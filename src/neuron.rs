use std::f64::consts::E;
use rand::Rng;

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

    // Derivative of the sigmoid function
    pub fn sigmoid_prime(x: f64) -> f64 {
        let sigmoid_x = Neuron::sigmoid(x);
        sigmoid_x * (1.0 - sigmoid_x)
    }

    // Performs the forward pass with sigmoid activation
    pub fn forward(&self, inputs: &[f64]) -> f64 {
        assert_eq!(inputs.len(), self.weights.len());
        let weighted_sum: f64 = inputs.iter().zip(self.weights.iter()).map(|(&input, &weight)| input * weight).sum();
        Neuron::sigmoid(weighted_sum + self.bias)
    }

    // Calculates the gradient of the neuron's output with respect to its inputs
    pub fn calculate_gradient(&self, inputs: &[f64], output_gradient: f64) -> (Vec<f64>, f64) {
        let weighted_sum: f64 = inputs.iter().zip(self.weights.iter()).map(|(&input, &weight)| input * weight).sum();
        let derivative = Neuron::sigmoid_prime(weighted_sum + self.bias);

        let weights_gradient: Vec<f64> = inputs.iter().map(|&input| input * derivative * output_gradient).collect();
        let bias_gradient = derivative * output_gradient;

        (weights_gradient, bias_gradient)
    }

    // Updates the weights and bias of the neuron
    pub fn update_parameters(&mut self, weights_gradient: &[f64], bias_gradient: f64, learning_rate: f64) {
        for (weight, gradient) in self.weights.iter_mut().zip(weights_gradient.iter()) {
            *weight -= learning_rate * gradient;
        }
        self.bias -= learning_rate * bias_gradient;
    }
}

// Example usage:
// let mut neuron = Neuron::new(3); // for 3 input connections
// let inputs = vec![0.5, -0.1, 0.8];
// let output = neuron.forward(&inputs);
// let (weights_gradient, bias_gradient) = neuron.calculate_gradient(&inputs, output_gradient);
// neuron.update_parameters(&weights_gradient, bias_gradient, 0.01);

