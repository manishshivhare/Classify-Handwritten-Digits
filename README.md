# MNIST Digit Recognizer Web Application

This web application allows users to draw digits (0-9) on a canvas and have them recognized by a machine learning model trained on the MNIST dataset.

## Features

- Interactive drawing canvas
- Real-time digit prediction
- Confidence score display
- Mobile-friendly with touch support

## Requirements

- Python 3.7+
- Flask
- TensorFlow
- NumPy
- Pillow (PIL)

## Installation

1. Clone this repository or download the files.

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python app.py
   ```

4. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

## How It Works

1. The application uses a Convolutional Neural Network (CNN) trained on the MNIST dataset.
2. When you draw a digit on the canvas, the application captures the drawing.
3. The drawing is processed to match the format expected by the model (28x28 grayscale image).
4. The model predicts which digit was drawn and provides a confidence score.

## Code Structure

- `app.py`: Main Flask application with routes and model training
- `templates/index.html`: HTML template with JavaScript for the drawing interface

## Model Architecture

The application uses a simple CNN with the following architecture:
- 2 convolutional layers with max pooling
- Flatten layer
- Dense hidden layer with ReLU activation
- Output layer with softmax activation for 10 classes (digits 0-9)

## License

This project is open-source and available for personal and educational use.