import subprocess
import re
import time

import cv2

def clean_ansi_escape(text):
    ansi_escape = re.compile(r'\x1b[^m]*m')
    return ansi_escape.sub('', text)

# Script de reconocimiento de placas
reconocimiento_placas_script = [
    'python',
    'C:/Users/Daniela/Desktop/CODIGOS BASE PYTHON/Deep_learning/Ejemplos_clase/semana 10/Deteccion_Placas/main.py'
    #'C:/Users/Daniela/Desktop/CODIGOS BASE PYTHON/Deep_learning/Ejemplos_clase/semana 10/Deteccion_Placas/main2.py'
]

# Ejecutar el script de reconocimiento de placas
result_placas = subprocess.run(reconocimiento_placas_script, capture_output=True, text=True)
output_placas = result_placas.stdout.strip()

print(f"Salida del script de reconocimiento de placas: {output_placas}")
print(f"Longitud de la salida del reconocimiento de placas: {len(output_placas)}")

# Script de reconocimiento facial
reconocimiento_facial_script = [
    'python',
    'C:/Users/Daniela/Desktop/CODIGOS BASE PYTHON/Deep_learning/Ejemplos_clase/semana 10/reconocimiento_facial_deep_class/main.py'
    #'C:/Users/Daniela/Desktop/CODIGOS BASE PYTHON/Deep_learning/Ejemplos_clase/semana 10/reconocimiento_facial_deep_class/main2.py'

]

# Ejecutar el script de reconocimiento facial utilizando Popen y communicate
reconocimiento_facial_process = subprocess.Popen(reconocimiento_facial_script, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
output_facial, _ = reconocimiento_facial_process.communicate()

output_facial = output_facial.strip()
# Limpiar el mensaje de error de la salida del reconocimiento facial
# Función para limpiar las secuencias de escape ANSI


# Limpiar la salida del reconocimiento facial
output_facial_cleaned = clean_ansi_escape(output_facial)


# Obtener los primeros 8 caracteres de la salida del reconocimiento facial
final_output_facial = clean_ansi_escape(output_facial)
final_output_facial = final_output_facial[:7]


# Obtener los primeros 8 caracteres de la salida del reconocimiento de placas
final_output_placas = clean_ansi_escape(output_placas)
final_output_placas = final_output_placas[2:-6]


if final_output_placas == final_output_facial:
    print('Usuario autorizado para entrada/salida')
    print('Abriendo compuertas...')
    time.sleep(1000)
    print('Compuertas Abiertas')
else:
    print('El usuario y la placa detectada no coinciden!')
    print('Acceso/Salida Denegada')