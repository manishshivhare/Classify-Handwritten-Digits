# app.py - Main Flask application

import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
import base64
from PIL import Image
import io
import os

app = Flask(__name__)

# Create directories for templates and static files
os.makedirs('templates', exist_ok=True)
os.makedirs('static', exist_ok=True)

# Function to load and train the model (similar to your original code)
def train_model():
    # Load MNIST dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normalize data
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    # Reshape data for the model
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    # Build a simple CNN model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train the model
    model.fit(x_train, y_train, epochs=5, verbose=1)
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_acc}')
    
    return model

# Load and train the model
model = train_model()

# Process the image and make a prediction
def process_image(image_data):
    # Convert base64 image to numpy array
    image_data = image_data.split(',')[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    
    # Convert to grayscale
    image = image.convert('L')
    
    # Resize to 28x28
    image = image.resize((28, 28))
    
    # Convert to numpy array and normalize
    img_array = np.array(image)
    img_array = img_array / 255.0
    
    # Invert image (as MNIST has white digits on black background)
    img_array = 1 - img_array
    
    # Reshape for model input
    img_array = img_array.reshape(1, 28, 28, 1)
    
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get image data from request
    image_data = request.json['image']
    
    # Process image
    processed_image = process_image(image_data)
    
    # Make prediction
    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction)
    confidence = float(prediction[0][predicted_digit])
    
    return jsonify({
        'digit': int(predicted_digit),
        'confidence': confidence
    })

if __name__ == '__main__':
    # Create the index.html file
    with open('templates/index.html', 'w') as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>MNIST Digit Recognizer</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            text-align: center;
        }
        #canvas-container {
            border: 2px solid #333;
            display: inline-block;
            margin: 20px 0;
        }
        canvas {
            cursor: crosshair;
            background-color: black;
        }
        button {
            margin: 10px;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
        }
        #result {
            font-size: 24px;
            margin: 20px 0;
            font-weight: bold;
        }
        .prediction-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 20px;
        }
        .prediction-box {
            border: 1px solid #ddd;
            padding: 15px;
            margin: 0 10px;
            border-radius: 5px;
            min-width: 150px;
        }
    </style>
</head>
<body>
    <h1>MNIST Digit Recognizer</h1>
    <p>Draw a digit (0-9) in the box below and click "Predict" to see the model's prediction.</p>
    
    <div id="canvas-container">
        <canvas id="drawing-canvas" width="280" height="280"></canvas>
    </div>
    
    <div>
        <button id="predict-btn">Predict</button>
        <button id="clear-btn">Clear</button>
    </div>
    
    <div class="prediction-container">
        <div class="prediction-box">
            <h3>Prediction</h3>
            <div id="result">-</div>
        </div>
        <div class="prediction-box">
            <h3>Confidence</h3>
            <div id="confidence">-</div>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('drawing-canvas');
        const ctx = canvas.getContext('2d');
        const predictBtn = document.getElementById('predict-btn');
        const clearBtn = document.getElementById('clear-btn');
        const resultDiv = document.getElementById('result');
        const confidenceDiv = document.getElementById('confidence');
        
        // Initialize canvas
        ctx.lineWidth = 15;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.strokeStyle = 'white';
        
        let isDrawing = false;
        
        // Drawing functionality
        canvas.addEventListener('mousedown', startDrawing);
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('mouseup', stopDrawing);
        canvas.addEventListener('mouseout', stopDrawing);
        
        // Touch support
        canvas.addEventListener('touchstart', function(e) {
            e.preventDefault();
            const touch = e.touches[0];
            const mouseEvent = new MouseEvent('mousedown', {
                clientX: touch.clientX,
                clientY: touch.clientY
            });
            canvas.dispatchEvent(mouseEvent);
        });
        
        canvas.addEventListener('touchmove', function(e) {
            e.preventDefault();
            const touch = e.touches[0];
            const mouseEvent = new MouseEvent('mousemove', {
                clientX: touch.clientX,
                clientY: touch.clientY
            });
            canvas.dispatchEvent(mouseEvent);
        });
        
        canvas.addEventListener('touchend', function(e) {
            e.preventDefault();
            const mouseEvent = new MouseEvent('mouseup');
            canvas.dispatchEvent(mouseEvent);
        });
        
        function startDrawing(e) {
            isDrawing = true;
            draw(e);
        }
        
        function draw(e) {
            if (!isDrawing) return;
            
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            ctx.beginPath();
            ctx.moveTo(lastX || x, lastY || y);
            ctx.lineTo(x, y);
            ctx.stroke();
            
            [lastX, lastY] = [x, y];
        }
        
        function stopDrawing() {
            isDrawing = false;
            [lastX, lastY] = [undefined, undefined];
        }
        
        clearBtn.addEventListener('click', function() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            resultDiv.textContent = '-';
            confidenceDiv.textContent = '-';
        });
        
        predictBtn.addEventListener('click', function() {
            // Get the image data from the canvas
            const imageData = canvas.toDataURL('image/png');
            
            // Send the image data to the server
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ image: imageData })
            })
            .then(response => response.json())
            .then(data => {
                resultDiv.textContent = data.digit;
                confidenceDiv.textContent = (data.confidence * 100).toFixed(2) + '%';
            })
            .catch(error => {
                console.error('Error:', error);
                resultDiv.textContent = 'Error';
            });
        });
    </script>
</body>
</html>
        """)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)