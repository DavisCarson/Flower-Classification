## Iris Flower Classification with Custom Neural Network

This project implements a simple neural network from scratch in Python to classify iris flowers into three species: *Iris-setosa*, *Iris-versicolor*, and *Iris-virginica*. The model is built with customizable input, hidden, and output layers and is trained using backpropagation with sigmoid activation.

### Key Features
- **Custom Neural Network Initialization**: Randomly initializes weights for hidden and output layers.
- **Activation and Backpropagation**: Uses sigmoid activation and its derivative for forward and backward passes.
- **Train and Predict Functions**: Includes `train` with backpropagation for weight updates and `predict` for classification.
- **Performance Evaluation**: Computes accuracy on a test dataset.

### Dataset
- Reads data from a text file, with attributes for sepal length, sepal width, petal length, and petal width.
- Shuffles and splits data into training and test sets.

### Training Configuration
- **Learning Rate**: 0.1
- **Epochs**: 1000

### Results
After training, the neural network outputs the accuracy of predictions on the test data.
