import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import numpy as np

# Directorio de las imágenes de entrenamiento
train_data_folder = "clientes"

# Listas para almacenar las características faciales y las etiquetas
X = []
y = []

# Crear instancias de MTCNN e InceptionResnetV1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=40,
    thresholds=[0.6, 0.7, 0.7], factor=0.9, post_process=True,
    device=device
)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Ajustar el LabelEncoder con las clases conocidas
for client_folder in os.listdir(train_data_folder):
    client_folder_path = os.path.join(train_data_folder, client_folder)
    if os.path.isdir(client_folder_path):
        X.append(client_folder_path)
        y.append(client_folder)

# Convertir las listas a matrices numpy
X = np.array(X)
y = np.array(y)

# Codificar las etiquetas
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Cargar el modelo SVM desde el archivo .pkl
model_filename_pkl = 'svm_model.pkl'
with open(model_filename_pkl, 'rb') as file:
    svm_classifier = pickle.load(file)

# Cargar el LabelEncoder desde el archivo .pkl
label_encoder_filename = 'label_encoder.pkl'
with open(label_encoder_filename, 'rb') as file:
    label_encoder = pickle.load(file)

# Ajustar el LabelEncoder con las clases conocidas
label_encoder.fit(y)


# Función para extraer características faciales
def extract_features(frame):
    try:
        x_aligned, prob = mtcnn(frame, return_prob=True)
        if x_aligned is None:
            pass

        x_aligned = torch.stack([x_aligned]).to(device)
        x_embed = model(x_aligned).detach().cpu()
        x_embed = x_embed.numpy()

        if x_embed is not None:
            return x_embed.ravel()
        else:
            pass
    except Exception as e:
        return None


# Iniciar la cámara
cap = cv2.VideoCapture(0)

# Variables para el seguimiento de predicciones
prediction_counter = 0
previous_prediction = None

face_predicted = False
while True:
    ret, frame = cap.read()

    try:
        boxes, _ = mtcnn.detect(frame)
    except Exception as e:

        boxes = None
        pass

    if boxes is not None:
        for i, box in enumerate(boxes):
            box = [int(coord) for coord in box]
            face = frame[box[1]:box[3], box[0]:box[2]]

            x_embed = extract_features(face)

            if x_embed is not None:
                prediction = svm_classifier.predict([x_embed])
                probabilidad = svm_classifier.predict_proba([x_embed])

                decoded_label = label_encoder.inverse_transform(prediction)
                probabilidad_predicha = probabilidad[0, prediction[0]]

                if probabilidad_predicha < 0.70:
                    prediction_label = 'Desconocido'

                    # Dibujar el cuadro del rostro y la etiqueta en el frame
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
                    cv2.putText(frame, prediction_label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (255, 0, 0), 2)

                else:
                    prediction_label = decoded_label[0]

                    print(prediction_label)
                    face_predicted = True
                    # Dibujar el cuadro del rostro y la etiqueta en el frame
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    cv2.putText(frame, prediction_label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0, 255, 0), 2)

    cv2.imshow('Reconocimiento Facial en Tiempo Real', frame)
    if face_predicted:
        cv2.waitKey(1000)
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()