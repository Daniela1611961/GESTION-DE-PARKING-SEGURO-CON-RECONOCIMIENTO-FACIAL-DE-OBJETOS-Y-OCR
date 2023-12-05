from facenet_pytorch import MTCNN
import torch
import numpy as np
import cv2
from PIL import Image
import os
from pymongo import MongoClient
from facenet_pytorch import InceptionResnetV1
import cv2
import numpy as np
import os
from datetime import datetime
import easyocr
reader = easyocr.Reader(['en'])

# Iniciar una conexión a MongoDB
client = MongoClient("localhost", 27017)
db = client["Clients_DB"]
clientes_collection = db["Clients"]
base_folder = "clientes"  # Carpeta base que contiene carpetas de clientes
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Running on device: {}'.format(device))

# crear instancia del detector de rostros MTCNN
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=40,#la imagen es de 160 px y se detectan rostros de minimo 40px
    thresholds=[0.6, 0.7, 0.7], factor=0.9, post_process=True,#atributos predeterminados
    device=device
)

# se crea el objeto camara que captura imagenes en tiempo real '0' es el indice de la camara
cam = cv2.VideoCapture(0)
# directorio local donde se encuentran alohadas las carpetas de imagenes de los clientes
destino = 'clientes/'

# Dimensionar las imagenes
new_dim = (300, 300)
continuar = True

# Definir funciones de carga y preprocesamiento de imágenes
def load_image(image_path):
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)#se convierte la imagen de BGR a RGB y es retornada por la funcion load_image(ruta)

def preprocess_image(image):
    return image / 255.0

def extract_features(frame):
    x_aligned, prob = mtcnn(frame, return_prob=True)
    x_embed = None  # Inicializar x_embed con None
    if x_aligned is not None:
        print('Rostro detectado con probabilidad: {:8f}'.format(prob))
        x_aligned = torch.stack([x_aligned]).to(device)
        x_embed = model(x_aligned).detach().cpu()
        x_embed = x_embed.numpy()

    # Comprobar si x_embed no es None antes de llamar a .ravel()
    if x_embed is not None:
        return x_embed.ravel()
    else:
        return None
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
# preguntar al usuario por el numero de fotos
fotos_por_cliente = 0
while fotos_por_cliente < 3:
    fotos_por_cliente = int(input("Número de fotos por cliente? Debe ser un numero mayor o igual a 3: "))

# se crea el directorio para cada cliente
while continuar:
    k = 0
    cliente = input("ID del cliente: ")
    path_cliente = destino + cliente  # se crea la ruta con el numero de ID
    if not os.path.exists(path_cliente):
        print("Creando el directorio para: '{}' ".format(path_cliente))
        os.makedirs(path_cliente)
    cliente_data = {
        "ID": cliente,
        "Nombre": input("Escriba nombre Completo: "),
        "edad": input("Edad: "),
        "caracteristicas_faciales": []
    }
    while k < fotos_por_cliente:
        retval, frame = cam.read()
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        boxes, confidence = mtcnn.detect(frame_pil)

        if np.ndim(boxes) != 0:
            box = boxes[0]
            c = confidence[0]
            box = box.astype(int)
            x, y, w, h = box

            if x > 0 and y > 0 and c > 0.95:
                cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
                cv2.imshow('frame', frame)
                print(f"Probabilidad de rostro: {c}")
                f_region = frame_pil.crop(box)
                f_region = f_region.resize(new_dim)
                new_name = path_cliente + '/img_' + str(k) + '.png'
                k = k + 1
                f_region.save(new_name)

        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cv2.destroyAllWindows()

    # Insertar los datos del cliente en la base de datos
    clientes_collection.insert_one(cliente_data)

    cont = input("¿Desea registrar otro cliente? (S/N): ")
    if cont.upper() == 'N':
        continuar = False

print('\nDatos guardados')

# Extraer características de todas las imágenes en las carpetas de clientes
for client_folder in os.listdir(base_folder):
    client_folder_path = os.path.join(base_folder, client_folder)
    if os.path.isdir(client_folder_path):
        print(f"Procesando imágenes en la carpeta del cliente: {client_folder}")
        for filename in os.listdir(client_folder_path):
            if filename.endswith(".png"):
                image_path = os.path.join(client_folder_path, filename)
                frame = cv2.imread(image_path)

                try:
                    x_embed = extract_features(frame)
                    print(f"Características extraídas de {filename}: {x_embed}")

                    # Insertar características en la base de datos Mongo
                    if x_embed is not None:
                        x_embed_list = x_embed.tolist()
                        clientes_collection.update_one(
                            {"ID": client_folder},
                            {"$push": {"caracteristicas_faciales": x_embed_list}}
                        )
                        print(f"Características insertadas en MongoDB para {client_folder}")

                except TypeError as e:
                    print(f"Error al procesar {filename}: {e}")

