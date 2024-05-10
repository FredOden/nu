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
    // Assuming the previous definitions for Neuron, Layer, and NeuralNetwork are available


    // Train the neural network with input data and expected output
    pub fn train(&mut self, inputs: Vec<Vec<f64>>, expected_outputs: Vec<Vec<f64>>, epochs: usize, learning_rate: f64) {
        for _ in 0..epochs {
            for (input, expected) in inputs.iter().zip(expected_outputs.iter()) {
                // Forward pass to get the output
                let outputs = self.forward(input.clone());

                // Calculate the loss and its gradient
                let loss_gradient = self.calculate_loss_gradient(&outputs, expected);

                // Backward pass to compute gradients
                self.backward(&loss_gradient);

                // Update weights and biases based on gradients
                self.update_parameters(learning_rate);
            }
        }
    }

    // Calculate the gradient of the loss function (mean squared error in this case)
    pub fn calculate_loss_gradient(&self, outputs: &Vec<f64>, expected: &Vec<f64>) -> Vec<f64> {
        outputs.iter().zip(expected.iter()).map(|(o, e)| 2.0 * (o - e)).collect()
    }

    // Backward pass to compute gradients for each layer
    pub fn backward(&mut self, loss_gradient: &Vec<f64>) {
        // This method should implement the backpropagation algorithm, which involves
        // computing the gradient of the loss with respect to each weight and bias
        // throughout the network layers.
        // ...
    }

    // Update weights and biases based on gradients
    pub fn update_parameters(&mut self, learning_rate: f64) {
        // This method should apply the computed gradients to update the network's
        // weights and biases, scaled by the learning rate.
        // ...
    }

}

