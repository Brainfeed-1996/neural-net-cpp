#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

// Sigmoid activation function
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of sigmoid
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

class Neuron {
public:
    double value;
    double activated_value;
    double derived_value;
    std::vector<double> weights;

    Neuron(int prev_layer_size) {
        for (int i = 0; i < prev_layer_size; i++) {
            weights.push_back(((double)rand() / RAND_MAX) * 2.0 - 1.0);
        }
    }
};

class Layer {
public:
    std::vector<Neuron> neurons;

    Layer(int size, int prev_layer_size) {
        for (int i = 0; i < size; i++) {
            neurons.push_back(Neuron(prev_layer_size));
        }
    }
};

int main() {
    srand(time(0));
    std::cout << "Initializing Neural Network..." << std::endl;
    
    // Example topology: 2 inputs, 3 hidden, 1 output
    Layer hidden_layer(3, 2);
    Layer output_layer(1, 3);

    std::cout << "Network initialized. Ready for training loops." << std::endl;
    return 0;
}
