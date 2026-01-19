from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import torch

from app.model import load_model
from app.utils import preprocess_image, postprocess

app = FastAPI(title="WikiArt Style Classifier")

device = "cpu"  # Docker-safe
model, idx_to_class = load_model(device)

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    input_tensor = preprocess_image(image, device)
    with torch.no_grad():
        output = model(input_tensor)

    predictions = postprocess(output, idx_to_class)

    return {
        "predictions": predictions
    }
