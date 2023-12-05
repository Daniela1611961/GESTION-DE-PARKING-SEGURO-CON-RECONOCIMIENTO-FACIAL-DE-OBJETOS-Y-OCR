import cv2
import numpy as np
from pymongo import MongoClient
import os
from datetime import datetime
import easyocr
reader = easyocr.Reader(['en'])

# Inicializar el objeto de captura de video con la ruta de la imagen
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error al abrir la camara")


# Conexión a MongoDB
client = MongoClient("localhost", 27017)
db = client["Clients_DB"]
clientes_collection = db["Clients"]



# Umbral de confianza y supresión no máxima
confThreshold = 0.1
nmsThreshold = 0.05

# Dimensiones de la entrada para el modelo YOLO
inpWidth = 416
inpHeight = 416

# Cargar nombres de clases para el modelo YOLO
classesFile = "yolo_utils/classes.names"

with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Cargar la configuración y pesos del modelo YOLO
modelConfiguration = r"yolo_utils/darknet-yolov3.cfg"
modelWeights = r"yolo_utils/lapi.weights"
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Obtener nombres de las capas de salida del modelo YOLO
def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

# Dibujar el cuadro delimitador predicho
def drawPred(classId, conf, left, top, right, bottom, frame):
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
    label = '%.2f' % conf
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (0, 0, 255), cv2.FILLED)
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)

# Eliminar las cajas delimitadoras con baja confianza usando supresión no máxima
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    cropped = None
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        bottom = top + height
        right = left + width
        cropped = frame[top:bottom, left:right].copy()
        drawPred(classIds[i], confidences[i], left, top, right, bottom, frame)
    if cropped is not None:
        return cropped

# Función para filtrar y tomar los primeros 6 caracteres alfanuméricos
def filter_and_take_first_six_chars(text):
    alphanumeric_chars = [char for char in text if char.isalnum()]
    return ''.join(alphanumeric_chars)[:6]

cliente_id = input("ID del cliente al que se asignarán las imágenes de la placa: ")
# Crear la carpeta para las imágenes de placas dentro de la carpeta del cliente
base_folder = "C:/Users/Daniela/Desktop/CODIGOS BASE PYTHON/Deep_learning/Ejemplos_clase/semana 10/reconocimiento_facial_deep_class/clientes"
path_cliente = os.path.join(base_folder, cliente_id)
path_cliente_placas = os.path.join(base_folder, cliente_id, "placas")
# Verificar si la carpeta del cliente ya existe
if os.path.exists(path_cliente):
    print(f"Advertencia: La carpeta del cliente {cliente_id} ya existe. No se creará una nueva carpeta.")
else:
    print(f"Advertencia: La carpeta del cliente {cliente_id} no está registrada. No se guardarán imágenes de placas.")
    exit()  # Salir del programa si la carpeta no existe

os.makedirs(path_cliente_placas, exist_ok=True)

# Contador para el número de fotos de placas capturadas
contador_fotos = 0
photos_per_placa = int(input("Ingrese el numero de fotos por placa"))


while cv2.waitKey(1) < 0 and contador_fotos <= photos_per_placa:
    hasFrame, frame = cap.read()
    if not hasFrame:
        print("Finalizado !!!")
        break
    blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
    net.setInput(blob)
    outs = net.forward(getOutputsNames(net))

    try:
        cropped = postprocess(frame, outs)

        # Verificar las dimensiones de la imagen antes de mostrar
        if cropped is not None and cropped.shape[0] > 0 and cropped.shape[1] > 0:
            # Guardar la imagen recortada localmente en la carpeta de placas del cliente
            img_filename = f"placa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            img_path = os.path.join(path_cliente_placas, img_filename)
            cv2.imwrite(img_path, cropped)
            contador_fotos += 1

            # Confirmar la guardado de la foto y la ruta
            print(f"Foto guardada en: {img_path}")
            # Realizar OCR en la imagen de la placa
            results = reader.readtext(cropped)
            all_texts = []

            # Almacenar resultados en la lista
            for (bbox, text, prob) in results:
                all_texts.append(text)

            # Unir todos los textos en una sola cadena sin comas ni espacios
            all_texts_str = ''.join(all_texts)

            # Omitir los espacios en la cadena resultante
            all_texts_str_no_spaces = ''.join(all_texts_str.split())

            placa_final = filter_and_take_first_six_chars(all_texts_str_no_spaces)
            # Imprimir la cadena resultante sin comas ni espacios
            print("Texto detectado en la placa:")
            print(placa_final)

            # Actualizar la base de datos MongoDB
            clientes_collection.update_one(
                {"ID": cliente_id},
                {"$push": {"imagenes de placa": placa_final}}
            )

        else:
            # Mostrar un mensaje en la cámara cuando no se detecta una placa
            cv2.putText(frame, "Esperando a detectar una placa...", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Video en Tiempo Real", frame)

    except cv2.error as e:
        print("Error al mostrar la imagen:", e)

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
