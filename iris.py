import random
import math

# Function for calculating Euclidean distance
def calculate_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))

# Initialize weights for the neural network with random values
def initialize_weights(num_inputs, num_hidden, num_outputs):
    hidden_layer_weights = [[random.uniform(-1, 1) for _ in range(num_inputs + 1)] for _ in range(num_hidden)]
    output_layer_weights = [[random.uniform(-1, 1) for _ in range(num_hidden + 1)] for _ in range(num_outputs)]
    return hidden_layer_weights, output_layer_weights

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Derivative of sigmoid activation function
def sigmoid_derivative(x):
    return x * (1 - x)

# Predict function for making predictions using the neural network
def predict(hidden_weights, output_weights, inputs):
    # Forward pass through hidden layer
    hidden_activations = [1]  # Bias for the hidden layer
    for weights in hidden_weights:
        activation = sum(weight * input_val for weight, input_val in zip(weights, [1] + inputs))
        hidden_activations.append(sigmoid(activation))

    # Forward pass through output layer
    output_activations = []
    for weights in output_weights:
        activation = sum(weight * hidden_val for weight, hidden_val in zip(weights, hidden_activations))
        output_activations.append(sigmoid(activation))

    return output_activations

# Train function using backpropagation
def train(hidden_weights, output_weights, inputs, targets, learning_rate):
    # Forward pass
    hidden_activations = [1]  # Bias for the hidden layer
    hidden_inputs = []  # Store inputs to hidden nodes for backprop
    for weights in hidden_weights:
        activation = sum(weight * input_val for weight, input_val in zip(weights, [1] + inputs))
        hidden_inputs.append(activation)
        hidden_activations.append(sigmoid(activation))

    output_activations = []
    output_inputs = []  # Store inputs to output nodes for backprop
    for weights in output_weights:
        activation = sum(weight * hidden_val for weight, hidden_val in zip(weights, hidden_activations))
        output_inputs.append(activation)
        output_activations.append(sigmoid(activation))

    # Backward pass
    output_errors = [target - output for target, output in zip(targets, output_activations)]
    output_deltas = [error * sigmoid_derivative(output) for error, output in zip(output_errors, output_activations)]

    hidden_errors = [sum(output_delta * weight for output_delta, weight in zip(output_deltas, [weights[i] for weights in output_weights])) for i in range(len(hidden_activations))]
    hidden_deltas = [hidden_error * sigmoid_derivative(hidden_activation) for hidden_error, hidden_activation in zip(hidden_errors, hidden_activations)]

    # Update output layer weights
    for i, output_delta in enumerate(output_deltas):
        for j, hidden_activation in enumerate(hidden_activations):
            output_weights[i][j] += learning_rate * output_delta * hidden_activation

    # Update hidden layer weights
    for i, hidden_delta in enumerate(hidden_deltas[1:]):  # Skip delta for bias
        for j, input_val in enumerate([1] + inputs):
            hidden_weights[i][j] += learning_rate * hidden_delta * input_val

    return hidden_weights, output_weights

# Create the neural network
num_inputs = 4  # Sepal length, sepal width, petal length, petal width
num_hidden = 5  # Number of hidden neurons
num_outputs = 3  # Iris-setosa, Iris-versicolor, Iris-virginica

# Initialize weights
hidden_layer_weights, output_layer_weights = initialize_weights(num_inputs, num_hidden, num_outputs)

# Load the dataset
with open('txt.txt', 'r') as file:
    data = [line.strip().split(',') for line in file.readlines()]
    data = [[float(x) for x in row[:-1]] + [row[-1]] for row in data]

# Shuffle and split the data
random.shuffle(data)
train_data = data[30:]
test_data = data[:30]

# Training parameters
learning_rate = 0.1
epochs = 1000

# Training loop
for epoch in range(epochs):
    for row in train_data:
        inputs = row[:-1]
        target_name = row[-1]
        targets = [1 if target_name == 'Iris-setosa' else 0, 1 if target_name == 'Iris-versicolor' else 0, 1 if target_name == 'Iris-virginica' else 0]
        hidden_layer_weights, output_layer_weights = train(hidden_layer_weights, output_layer_weights, inputs, targets, learning_rate)

# Evaluate the network
correct = 0
for row in test_data:
    inputs = row[:-1]
    target_name = row[-1]
    outputs = predict(hidden_layer_weights, output_layer_weights, inputs)
    predicted = outputs.index(max(outputs))
    actual = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'].index(target_name)
    if predicted == actual:
        correct += 1

accuracy = correct / len(test_data) * 100
print(f'Accuracy: {accuracy}%')
