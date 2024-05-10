use crate::neuron;

pub struct Layer {
    neurons: Vec<neuron::Neuron>,
}

impl Layer {
    // Create a new layer with a specified number of neurons, each with a specified number of inputs
    pub fn new(neuron_count: usize, input_size: usize) -> Self {
        let neurons = (0..neuron_count).map(|_| neuron::Neuron::new(input_size)).collect();
        Layer { neurons }
    }

    // Perform a forward pass through the layer
    pub fn forward(&self, inputs: &[f64]) -> Vec<f64> {
        self.neurons.iter().map(|neuron| neuron.forward(inputs)).collect()
    }
}

// Example usage:
// let layer = Layer::new(5, 3); // for 5 neurons, each with 3 input connections
// let inputs = vec![0.5, -0.1, 0.8];
// let outputs = layer.forward(&inputs);

