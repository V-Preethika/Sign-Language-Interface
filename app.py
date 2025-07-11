from flask import Flask, render_template, request, jsonify
import os
import cv2
import subprocess

app = Flask(__name__)

DATA_DIR = './data'

# Create the data directory if it doesn't exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture/<class_name>', methods=['POST'])
def capture_images(class_name):
    if class_name not in ['A', 'B', 'C']:
        return jsonify({"error": "Invalid class name"}), 400

    # Call the image collection script
    command = f"python collect_imgs.py {class_name}"
    subprocess.Popen(command, shell=True)
    
    return jsonify({"message": f"Started capturing images for class {class_name}"}), 200

@app.route('/create_dataset', methods=['POST'])
def create_dataset():
    # Call the dataset creation script
    command = "python create_dataset.py"
    subprocess.Popen(command, shell=True)
    
    return jsonify({"message": "Started creating dataset"}), 200

@app.route('/train_classifier', methods=['POST'])
def train_classifier():
    # Call the training script
    command = "python train_classifier.py"
    subprocess.Popen(command, shell=True)
    
    return jsonify({"message": "Started training classifier"}), 200

@app.route('/run_inference', methods=['POST'])
def run_inference():
    # Call the inference script
    command = "python inference_classifier.py"
    subprocess.Popen(command, shell=True)
    
    return jsonify({"message": "Started running inference"}), 200

if __name__ == '__main__':
    app.run(debug=True)
