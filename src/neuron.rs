use rand::Rng;

pub struct Neuron {
    weights: Vec<f64>, // A vector of weights for the inputs
    bias: f64,         // The bias term
}

impl Neuron {
    // A method to create a new neuron with random weights and bias
    pub fn new(input_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights = (0..input_size).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let bias = rng.gen_range(-1.0..1.0);

        Neuron { weights, bias }
    }

    // A method for the forward pass which takes inputs and produces an output
    pub fn forward(&self, inputs: &[f64]) -> f64 {
        assert_eq!(inputs.len(), self.weights.len());
        inputs.iter().zip(self.weights.iter()).map(|(&input, &weight)| input * weight).sum::<f64>() + self.bias
    }
}

