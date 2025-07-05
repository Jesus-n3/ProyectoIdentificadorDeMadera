from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from io import BytesIO
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

# Cargar CSV y etiquetas
csv_path = "wood_db.imagenes.csv"
df = pd.read_csv(csv_path)
le = LabelEncoder()
df['label'] = le.fit_transform(df['tipo'])
class_names = le.classes_

# Modelo
num_classes = len(class_names)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("model.pt", map_location=device))
model.eval()
model.to(device)

# Transformaciones
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            label = class_names[predicted.item()]

        return JSONResponse(content={"tipo": label})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
def root():
    return {"message": "API de Clasificación de Maderas - Usa POST /predict con una imagen"}
