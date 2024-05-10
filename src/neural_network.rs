use crate::layer;

pub struct NeuralNetwork {
    layers: Vec<layer::Layer>,
}

impl NeuralNetwork {
    // Create a new neural network with a vector of layers
    pub fn new(layers: Vec<layer::Layer>) -> Self {
        NeuralNetwork { layers }
    }

    // Perform a forward pass through the entire network
    pub fn forward(&self, input: Vec<f64>) -> Vec<f64> {
        self.layers.iter().fold(input, |acc, layer| layer.forward(&acc))
    }
}

