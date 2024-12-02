from flask import Flask, request, jsonify, render_template
import pandas as pd

app = Flask(__name__)

# Cargar documentos y vectorizadores
reuters_dir = r"..\data\reuters"
input_excel_path = os.path.join(reuters_dir, "reuters_data_preprocessed.xlsx")
df = pd.read_excel(input_excel_path)

# Vectorizador y matriz TF-IDF preentrenados
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Contenido Preprocesado'].fillna(""))

@app.route('/')
def index():
    """Página principal con el formulario de búsqueda."""
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """Procesa la consulta del usuario y devuelve resultados."""
    query = request.form.get('query', '')
    top_k = int(request.form.get('top_k', 10))
    
    # Procesar consulta y buscar
    query_vector = tfidf_vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    ranked_indices = cosine_similarities.argsort()[-top_k:][::-1]
    
    # Generar resultados
    results = []
    for i in ranked_indices:
        results.append({
            "Documento": df.iloc[i]['Nombre'],
            "Similitud": round(cosine_similarities[i], 4),
            "Contenido": df.iloc[i]['Contenido'][:200] + "..."
        })
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
