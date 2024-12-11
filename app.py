import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, render_template, request, redirect, flash, url_for
import numpy as np

app = Flask(__name__)
app.secret_key = 'brain_hemorrhage_detection'

# Load the trained model
MODEL_PATH = "resnet18_brain_ct_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model architecture (ResNet18)
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image).item()
        prediction = "Hemorrhage" if output > 0.5 else "Normal"
        probability = torch.sigmoid(torch.tensor(output)).item()
    
    return prediction, probability

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file uploaded')
        return redirect(url_for('home'))

    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('home'))

    if file:
        file_path = os.path.join('static/uploads', file.filename)
        file.save(file_path)

        prediction, probability = predict_image(file_path)
        return render_template('index.html', prediction=prediction, probability=probability, image_path=file_path)

if __name__ == '__main__':

    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
