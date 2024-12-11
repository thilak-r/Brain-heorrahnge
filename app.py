import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
from flask import Flask, render_template, request, redirect, flash, jsonify

app = Flask(__name__)
app.secret_key = 'brain_hemorrhage_detection'

# Load the trained model
MODEL_PATH = "resnet18_brain_ct_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model architecture (ResNet18)
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 1)  # Binary classification
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
    try:
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image).item()
            probability = torch.sigmoid(torch.tensor(output)).item()
            prediction = "Hemorrhage" if probability > 0.5 else "Normal"
        
        return prediction, probability
    except UnidentifiedImageError:
        return "Invalid Image", None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    if file:
        upload_folder = 'static/uploads'
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)

        prediction, probability = predict_image(file_path)
        if probability is None:
            return jsonify({'error': 'Invalid image file'})

        return jsonify({'prediction': prediction, 'probability': round(probability, 2), 'image_path': f'/{file_path}'})

if __name__ == '__main__':

    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
