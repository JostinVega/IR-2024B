import pandas as pd
from pymongo import MongoClient
from urllib.parse import quote_plus

# Función principal para cargar datos a MongoDB Atlas
def cargar_datos_a_mongodb(excel_file, database_name, collection_name, mongo_uri):
    """
    Carga datos de un archivo Excel a una colección de MongoDB Atlas.

    Parámetros:
    - excel_file: Ruta al archivo Excel (str).
    - database_name: Nombre de la base de datos en MongoDB (str).
    - collection_name: Nombre de la colección donde se almacenarán los datos (str).
    - mongo_uri: URI de conexión de MongoDB Atlas (str).
    """
    # Leer el archivo Excel
    print("Leyendo el archivo Excel...")
    df = pd.read_excel(excel_file, usecols=["Nombre", "Titulo", "Titulo Preprocesado", "Contenido", "Contenido Preprocesado Sin Stemming", "Contenido Preprocesado", "Origen", "Categoria"])

    # Renombrar columnas
    df.rename(columns={
        "Titulo Preprocesado": "Titulo_Preprocesado",
        "Contenido Preprocesado": "Contenido_Preprocesado"
    }, inplace=True)

    # Convertir a lista de diccionarios
    data = df.to_dict(orient='records')

    # Conectarse a MongoDB Atlas
    print("Conectándose a MongoDB Atlas...")
    client = MongoClient(mongo_uri)
    db = client[database_name]
    collection = db[collection_name]

    # Insertar los datos en la colección
    print("Subiendo datos a la colección...")
    collection.insert_many(data)

    print("Datos subidos exitosamente.")

# Parámetros de conexión y archivo
if __name__ == "__main__":
    # Ruta al archivo Excel
    excel_file = r"..\reute\reuters\reuters_data_preprocessed.xlsx"

    # Credenciales y configuración de MongoDB Atlas
    usuario = "jostinvega"
    password = "JAvm1402@"  # Cambia esto a tu contraseña
    encoded_user = quote_plus(usuario)
    encoded_password = quote_plus(password)

    mongo_uri = f"mongodb+srv://{encoded_user}:{encoded_password}@cluster0.jet01gs.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

    database_name = "reuters"
    collection_name = "corpus"

    # Llamar a la función
    cargar_datos_a_mongodb(excel_file, database_name, collection_name, mongo_uri)
