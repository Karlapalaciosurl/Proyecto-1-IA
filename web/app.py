from flask import Flask, render_template, request, jsonify
import pickle
import re
import math
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import uuid

nltk.download('stopwords', quiet=True)

app = Flask(__name__)

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Cargar modelo
with open("../modelo/modelo_naive_bayes.pkl", "rb") as f:
    modelo = pickle.load(f)

prob_prior = modelo["prob_prior"]
prob_palabras = modelo["prob_palabras"]
vocabulario = modelo["vocabulario"]

def preprocesar(texto):
    texto = re.sub(r'\{\{.*?\}\}', '', texto)
    texto = texto.lower()
    texto = re.sub(r'[^a-z\s]', '', texto)
    tokens = texto.split()
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens

def predecir(texto):
    tokens = preprocesar(texto)
    scores = {}
    for c in prob_prior:
        score = prob_prior[c]
        for token in tokens:
            if token in vocabulario:
                score += prob_palabras[c][token]
        scores[c] = score
    return max(scores, key=scores.get)

@app.route("/")
def inicio():
    ticket_id = "TKT-" + str(uuid.uuid4())[:8].upper()
    return render_template("index.html", ticket_id=ticket_id)

@app.route("/clasificar", methods=["POST"])
def clasificar():
    datos = request.json
    texto = datos.get("descripcion", "")
    categoria = predecir(texto)
    return jsonify({"categoria": categoria})

if __name__ == "__main__":
    app.run(debug=True)