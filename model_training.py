import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
import matplotlib.pyplot as plt

# Ruta de la carpeta principal de H
root_dir = "data"
train_dir = os.path.join(root_dir, 'train')
val_dir = os.path.join(root_dir, 'val')

# Tamaño común para redimensionar las imágenes
common_size = (75, 75)

# Listas para almacenar imágenes y etiquetas
X_train = []
y_train = []
X_val = []
y_val = []


def load_and_resize_image(img_path):
    img = Image.open(img_path)
    img = img.resize(common_size)
    img = np.array(img)

    # Asegurar que la imagen tenga 3 canales
    if len(img.shape) == 2:  # Si es en escala de grises, convierte a RGB
        img = np.stack((img,) * 3, axis=-1)
    elif img.shape[-1] > 3:  # Si tiene más de 3 canales, toma solo los primeros 3
        img = img[:, :, :3]

    return img


# Cargar imágenes de entrenamiento
for class_name in os.listdir(train_dir):
    class_dir = os.path.join(train_dir, class_name)
    for img_file in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_file)
        X_train.append(img_path)
        y_train.append(class_name)

# Convertir a matrices de NumPy
X_train = np.array(X_train, dtype=object)  # Cambiado a object para almacenar rutas de archivo
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)

# Convertir matrices de NumPy a tensores de PyTorch
y_train = torch.tensor(y_train, dtype=torch.long)

# Definir transformaciones de datos
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

# Crear conjuntos de datos y dataloaders
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]  # Obtener la ruta de la imagen
        image = load_and_resize_image(img_path)

        if self.transform:
            image = self.transform(image)

        label = self.labels[index]

        return image, label

train_dataset = CustomDataset(image_paths=X_train, labels=y_train, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Crear conjuntos de datos y dataloaders para conjunto de validación
X_val = []  # Agregar rutas de archivo para imágenes de validación
y_val = []  # Agregar etiquetas para imágenes de validación

for class_name in os.listdir(val_dir):
    class_dir = os.path.join(val_dir, class_name)
    for img_file in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_file)
        X_val.append(img_path)
        y_val.append(class_name)

# Convertir a matrices de NumPy
X_val = np.array(X_val, dtype=object)  # Cambiado a object para almacenar rutas de archivo
label_encoder = LabelEncoder()
y_val = label_encoder.fit_transform(y_val)

# Convertir matrices de NumPy a tensores de PyTorch
y_val = torch.tensor(y_val, dtype=torch.long)
# Crear conjuntos de datos y dataloaders para conjunto de entrenamiento

val_dataset = CustomDataset(image_paths=X_val, labels=y_val, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

# Definir el modelo
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.base_model = models.vgg16(pretrained=True)
        self.base_model.features[0] = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.base_model.classifier[6] = nn.Linear(4096, 36)

    def forward(self, x):
        return self.base_model(x)

# Crear modelo, función de pérdida y optimizador
model = CustomModel()
criterion = nn.CrossEntropyLoss()  # Cambiado a CrossEntropyLoss para clasificación multiclase
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Entrenamiento del modelo
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Crear una lista para almacenar las pérdidas promedio por época
val_losses_epoch = []

for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calcular y almacenar la pérdida promedio por época
    val_losses_epoch.append(val_loss / len(val_loader))
    print(
        f"Epoch {epoch + 1}/{num_epochs} - Validation Loss: {val_loss / len(val_loader)} - Validation Accuracy: {100 * correct / total}%")


# Plotear las pérdidas después de que se hayan recopilado
plt.plot(range(1, num_epochs + 1), val_losses_epoch)

# Evaluación del modelo
model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
# Imprimir informes de clasificación y matriz de confusión
print(classification_report(all_labels, all_predictions))
cm = confusion_matrix(all_labels, all_predictions)
print("Confusion Matrix:")
print(cm)

# Guardar el modelo
torch.save(model.state_dict(), 'model_HandWrite.pth')

# Visualización de la precisión del entrenamiento
plt.figure()
plt.plot(range(1, num_epochs + 1), val_losses_epoch)
plt.title("Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()