from pymongo import MongoClient

#crear una conexion a Mongo DB
client = MongoClient("localhost", 27017)

#crear base de datos llamada clientes_db
db = client["Clients_DB"]

#crear una coleccion llamada clientes
clientes_collection = db["Clients"]

# Datos de un cliente
cliente = {
    "nombre": "Nombre del Cliente",
    "edad": 30,
    "identificacion": "Numero de Identificacion",
    "caracteristicas_faciales": [

    ],
    "imagenes de placa": [

    ]
}

# Insertar el cliente en la colección para que se cree la base de datos# este cliente se puede eliminar despues
resultado = clientes_collection.insert_one(cliente)

