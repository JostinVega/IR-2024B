from flask import Flask, request, render_template, jsonify
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
from urllib.parse import quote_plus
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv
import string
import os

# Descargar recursos necesarios de NLTK
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
CORS(app, resources={r"/search": {"origins": "http://localhost:3000"}})

# Credenciales y configuración de MongoDB Atlas
# Cargar variables del archivo .env
load_dotenv()

# Obtener valores de las variables de entorno
usuario = os.getenv("MONGO_USER")
password = os.getenv("MONGO_PASSWORD")
cluster = os.getenv("MONGO_CLUSTER")
app_name = os.getenv("MONGO_APP_NAME")

# Codificar usuario y contraseña
encoded_user = quote_plus(usuario)
encoded_password = quote_plus(password)

# Construir la URI
mongo_uri = f"mongodb+srv://{encoded_user}:{encoded_password}@{cluster}/?retryWrites=true&w=majority&appName={app_name}"

client = MongoClient(mongo_uri)
db = client['reuters']
collection = db['corpus']

# Cargar índice invertido desde un archivo Excel
def cargar_indice_invertido(ruta_indice="inverted_index.xlsx"):
    """Cargar índice invertido desde un archivo Excel."""
    return pd.read_excel(ruta_indice)

indice_invertido = cargar_indice_invertido()

# Recuperar datos de MongoDB
def cargar_datos_desde_mongodb():
    """Cargar datos desde la base de datos MongoDB."""
    documentos = list(collection.find({}, {
        "_id": 0,
        "Nombre": 1,
        "Titulo": 1,
        "Contenido": 1,
        "Contenido_Preprocesado": 1,
    }))
    return documentos

# Función para limpiar texto
def clean_text(text):
    """Limpia y normaliza el texto."""
    text = text.lower()  # Convertir a minúsculas
    text = text.translate(str.maketrans('', '', string.punctuation))  # Eliminar puntuación
    text = text.strip()  # Eliminar espacios iniciales y finales
    return text

# Función de preprocesamiento sin stemming
def preprocess_text_without_stemming(content):
    """Realiza limpieza, tokenización, eliminación de stopwords y normalización."""
    # Limpieza del texto
    cleaned_text = clean_text(content)
    # Tokenización
    tokens = word_tokenize(cleaned_text)
    # Eliminación de stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Reconstrucción del texto preprocesado
    preprocessed_text = " ".join(tokens)
    return preprocessed_text

# Vectorización TF-IDF
def vectorizar_tfidf(documentos):
    corpus = [doc['Contenido_Preprocesado'] for doc in documentos]
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    return tfidf_vectorizer, tfidf_matrix

# Vectorización Bag of Words
def vectorizar_bow(documentos):
    corpus = [doc['Contenido_Preprocesado'] for doc in documentos]
    bow_vectorizer = CountVectorizer()
    bow_matrix = bow_vectorizer.fit_transform(corpus)
    return bow_vectorizer, bow_matrix

# Función para convertir texto en un vector promedio usando Word2Vec
def get_average_word2vec_vector(text, model):
    words = text.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if len(word_vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(word_vectors, axis=0)

# Obtener documentos relevantes desde el índice invertido
def obtener_documentos_relevantes(query_terms, indice_invertido):
    relevant_docs = set()
    for term in query_terms:
        term_docs = indice_invertido[indice_invertido["Término"] == term]["Documentos"]
        if term_docs.empty:
            print(f"Término no encontrado en el índice invertido: {term}")  # Depuración
        else:
            for doc_list in term_docs:
                try:
                    doc_ids = list(map(int, doc_list.split(',')))
                    relevant_docs.update(doc_ids)
                except ValueError:
                    continue
    return relevant_docs

# Cargar y vectorizar datos al inicio
documentos = cargar_datos_desde_mongodb()
tfidf_vectorizer, tfidf_matrix = vectorizar_tfidf(documentos)
bow_vectorizer, bow_matrix = vectorizar_bow(documentos)

corpus = [doc["Contenido_Preprocesado"] for doc in documentos]
tokenized_corpus = [doc.split() for doc in corpus]

# Entrenar modelo Word2Vec
word2vec_model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1)
corpus_vectors = [get_average_word2vec_vector(doc, word2vec_model) for doc in corpus]

@app.route('/')
def home():
    """Página principal."""
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """Procesar consulta y devolver resultados."""
    try:
        data = request.get_json()
        if not data or "query" not in data:
            return jsonify({"error": "No se recibió ninguna consulta"}), 400
        
        query = data.get("query", "").strip()
        if not isinstance(query, str) or query == "":
            return jsonify({"error": "La consulta debe ser un string válido"}), 400
        
        method = data.get('method', 'tfidf')

        # Preprocesar la consulta
        preprocessed_query = preprocess_text_without_stemming(query)  # MISMO PREPROCESAMIENTO

        # Vectorización y cálculo de similitud
        if method == 'tfidf':
            query_vector = tfidf_vectorizer.transform([preprocessed_query])
            if query_vector.nnz == 0:  # Verificar si el vector tiene valores no nulos
                return jsonify({"error": "La consulta no tiene términos relevantes en el vocabulario"}), 400
            similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        elif method == 'bow':
            query_vector = bow_vectorizer.transform([preprocessed_query])
            if query_vector.nnz == 0:
                return jsonify({"error": "La consulta no tiene términos relevantes en el vocabulario"}), 400
            similarities = cosine_similarity(query_vector, bow_matrix).flatten()
        elif method == 'word2vec':
            query_vector = get_average_word2vec_vector(preprocessed_query, word2vec_model)
            similarities = cosine_similarity([query_vector], corpus_vectors).flatten()
        else:
            return jsonify({"error": "Método de búsqueda no válido"}), 400

        # Seleccionar documentos recuperados (similitud > 0)
        retrieved_indices = [idx for idx, sim in enumerate(similarities) if sim > 0]
        retrieved_docs = set(idx for idx in retrieved_indices)

        # Obtener documentos relevantes del índice invertido
        query_terms = preprocessed_query.split()  # Términos preprocesados
        true_positive_docs = obtener_documentos_relevantes(query_terms, indice_invertido)

        # Calcular métricas
        if retrieved_docs and true_positive_docs:
            true_positives = len(retrieved_docs & true_positive_docs)
            false_positives = len(retrieved_docs - true_positive_docs)
            false_negatives = len(true_positive_docs - retrieved_docs)

            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        else:
            precision = recall = f1_score = 0

        # Preparar resultados
        results = []
        for idx in retrieved_indices:
            results.append({
                "Nombre": documentos[idx]['Nombre'],
                "Similitud": round(float(similarities[idx]), 4),
                "Titulo": documentos[idx]['Titulo'],
                "Contenido": documentos[idx]['Contenido'][:500]
            })

        response = {
            "numero_resultados": len(results),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1_score, 4),
            "resultados": results
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
