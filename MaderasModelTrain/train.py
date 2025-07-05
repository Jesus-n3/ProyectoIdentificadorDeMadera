import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import requests
from io import BytesIO
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Cargar CSV
df = pd.read_csv("wood_db.imagenes.csv")

# Codificar etiquetas
le = LabelEncoder()
df['label'] = le.fit_transform(df['tipo'])

# Transformaciones
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dataset personalizado
class WoodDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        url = self.dataframe.iloc[idx]['url']
        label = self.dataframe.iloc[idx]['label']

        response = requests.get(url, timeout=30)
        image = Image.open(BytesIO(response.content)).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

# Separar datos en entrenamiento y validación
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

train_dataset = WoodDataset(train_df, transform=transform)
val_dataset = WoodDataset(val_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Modelo
num_classes = df['label'].nunique()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Entrenamiento
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

print("\U0001F680 Entrenando modelo...")
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"\U0001F4DA Epoch {epoch+1}/{num_epochs} - Loss: {running_loss/len(train_loader):.4f}")

# Validación
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"✅ Precisión en validación: {accuracy:.2f}%")

# Guardar modelo
torch.save(model.state_dict(), "model.pt")
print("💾 Modelo guardado en model.pt")
