import torch
import pickle
from torch import nn
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Loading the trained model
class CropModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CropModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.selu(self.fc1(x))
        x = torch.selu(self.fc2(x))
        x = torch.selu(self.fc3(x))
        x = self.fc4(x)
        return x

input_size = 7  
num_classes = 22
model = CropModel(input_size, num_classes)
model.load_state_dict(torch.load("./model/baseline/baseline.hdf5"))
model.eval()  # Set the model to evaluation mode

# Load label encoder and normalization parameters
with np.load('./model/normalization/normalization.npz') as data:
    mean = torch.tensor(data['mean'])
    std = torch.tensor(data['std'])
with open("./model/pkl_files/encoder.pkl", "rb") as file:
    encoder = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        features = np.array([float(data['N']), float(data['P']), float(data['K']),
                             float(data['temperature']), float(data['humidity']),
                             float(data['ph']), float(data['rainfall'])])
        
        # Normalize features
        features = (torch.tensor(features, dtype=torch.float32) - mean) / std
        features = features.view(1, -1)  # Reshape for model input

        # Get prediction
        with torch.no_grad():
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            crop_prediction = encoder.inverse_transform([predicted.item()])[0]

        return jsonify({'crop': crop_prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)