# Bidirectional LSTM for Regression

This repository contains the implementation of a **Bidirectional Long Short-Term Memory (LSTM)** model for regression using Keras. The dataset is a CSV file containing 16 features and a target label. The model is designed to predict the target label based on the features.

## Project Structure

- **`bidirectional_lstm_regression.py`**: Python script containing the code to load the dataset, preprocess the data, build the Bidirectional LSTM model, and train it.
- **`dataset.csv`**: Sample dataset used for training the model.

## Dataset
- The dataset contains 16 features and 1 target label.
- The dataset is loaded as a Pandas DataFrame and converted to a NumPy array for processing.

## Model Architecture
The model uses the following architecture:
- **Bidirectional LSTM**: Input layer with 5 units.
- **Dense Layers**: Three fully connected layers with 10, 100, and 1000 neurons, using `tanh` activation.
- **Output Layer**: A single neuron for regression output.

## Requirements

To run this project, you will need the following libraries:
- Python 3.x
- Keras
- TensorFlow
- Pandas
- Scikit-learn
- NumPy

Install the required packages using pip:
```bash
pip install keras tensorflow pandas scikit-learn numpy
