import sklearn.calibration
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import pickle

# Directorio de las imágenes de entrenamiento
train_data_folder = "clientes"

# Listas para almacenar las características faciales y las etiquetas
X = []
y = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)#transferencia de aprendizaje


#se crea instancia del detector de rostros
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=40,#la imagen es de 160 px y se detectan rostros de minimo 20px
    thresholds=[0.6, 0.7, 0.7], factor=0.98, post_process=True,#atributos predeterminados
    device=device
)
#funcion para extraer el vector de 512 elementos que refleja las caracteristicas faciales de cada cliente
def extract_features(frame):#funcion que extrae las caracteristicas
    x_aligned, prob = mtcnn(frame, return_prob=True)#objeto retorna rostro detectado y probabilidad
    x_embed = None  # Inicializar x_embed con None
    if x_aligned is not None:#si hay un rostro detectado por como minimo
        print('Rostro detectado con probabilidad: {:8f}'.format(prob))
        x_aligned = torch.stack([x_aligned]).to(device)
        x_embed = model(x_aligned).detach().cpu()
        x_embed = x_embed.numpy()
        return x_embed.ravel()

    # Comprobar si hay
    else:
        return None

# Recorrer las carpetas de clientes
for client_folder in os.listdir(train_data_folder):#por cada carpeta de cliente en la carpeta de datos de entrenamiento
    client_folder_path = os.path.join(train_data_folder, client_folder)#la ruta del cliente es la ruta de la carpeta de datos de entrenamiento + la ruta de la carpeta del cliente
    if os.path.isdir(client_folder_path):#si existe la ruta
        for filename in os.listdir(client_folder_path):#por cada archivo en la carpeta de cada cliente
            if filename.endswith(".png"):#si la imagen esta en formato png
                image_path = os.path.join(client_folder_path, filename)#la ruta de la imagen es la suma de la ruta de los datos de entrenamiento + ruta de carpeta cleinte + ruta de la imagen
                frame = cv2.imread(image_path)#se define el frame de cada archivo en la carpeta de cada cliente

                try:#intentar
                    x_embed = extract_features(frame)#extraer caracteristicas faciales del frame
                    label = client_folder#se define la etiqueta como el nombre de la carpeta del cliente
                    print(label)# imprimir etiqueta
                    if x_embed is not None:#si se detecta un rostro
                        X.append(x_embed)#se agrega a los datos de entrenamiento
                        y.append(client_folder)#se agrega la etiqueta
                except TypeError as e:
                    print(f"Error al procesar {filename}: {e}")

# Convertir las listas a matrices numpy
X = np.array(X)
y = np.array(y)

#crear un codificador de etiquetas
label_encoder = LabelEncoder()

# Codificar las etiquetas
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Guardar el LabelEncoder en un archivo .pkl
label_encoder_filename = 'label_encoder.pkl'
with open(label_encoder_filename, 'wb') as file:
    pickle.dump(label_encoder, file)

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Entrenar un clasificador SVM
svm = SVC()
svm_classifier = sklearn.calibration.CalibratedClassifierCV(svm)

#Ajustar el clasificador SVM
svm_classifier.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred = svm_classifier.predict(X_test)

# Calcular la precisión
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del clasificador SVM: {accuracy * 100:.2f}%')

# Visualizar la reducción de dimensionalidad con PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Guardar el modelo en un archivo .pkl
model_filename_pkl = 'svm_model.pkl'
with open(model_filename_pkl, 'wb') as file:
    pickle.dump(svm_classifier, file)

print(f'Modelo SVM guardado como {model_filename_pkl}')

# Graficar las características faciales en un espacio bidimensional
plt.figure(figsize=(10, 8))
for label in np.unique(y):
    mask = (y == label)
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=label, alpha=0.7)

plt.title('Reducción de dimensionalidad con PCA')
plt.xlabel('Componente principal 1')
plt.ylabel('Componente principal 2')
plt.legend()
plt.show()
