from flask import Flask, request, render_template, jsonify
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Cargar datos procesados
reuters_dir = r"..\data\reuters"
input_excel_path = os.path.join(reuters_dir, "reuters_data_preprocessed.xlsx")
df = pd.read_excel(input_excel_path)

# Vectorización TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Contenido Preprocesado'].fillna(""))

@app.route('/')
def home():
    """Página principal."""
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """Procesar consulta y devolver resultados."""
    query = request.json.get('query', '')
    top_k = int(request.json.get('top_k', 10))
    
    # Vectorizar consulta y calcular similitud coseno
    query_vector = tfidf_vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Seleccionar los top_k resultados
    ranked_indices = cosine_similarities.argsort()[-top_k:][::-1]
    
    # Preparar resultados
    results = []
    for idx in ranked_indices:
        results.append({
            "Nombre": df.iloc[idx]['Nombre'],
            "Similitud": round(cosine_similarities[idx], 4),
            "Contenido": df.iloc[idx]['Contenido'][:200] + "..."  # Solo muestra los primeros 200 caracteres
        })
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
