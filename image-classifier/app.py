from flask import Flask, render_template, request
from torchvision import models
from PIL import Image
import torch

app = Flask(__name__)

# Load pre-trained ResNet-18
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.eval()

# Image preprocessing
preprocess = weights.transforms()

# ImageNet class names
categories = weights.meta["categories"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    image = request.files["image"]

    # Open image
    img = Image.open(image).convert("RGB")

    # Prepare image
    img = preprocess(img).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output = model(img)

    # Convert output to probabilities
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get top 3 predictions
    top3 = torch.topk(probabilities, 3)

    predictions = []

    for i in range(3):
        index = top3.indices[i].item()
        confidence = top3.values[i].item() * 100

        predictions.append({
            "name": categories[index],
            "confidence": round(confidence, 2)
        })

    return render_template(
        "result.html",
        predictions=predictions
    )


if __name__ == "__main__":
    app.run(debug=True)
    