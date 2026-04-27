import csv
import re
import math
import random
import pickle
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocesar(texto):
    texto = re.sub(r'\{\{.*?\}\}', '', texto)
    texto = texto.lower()
    texto = re.sub(r'[^a-z\s]', '', texto)
    tokens = texto.split()
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens

archivo = "../Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"

with open(archivo, encoding="utf-8") as f:
    lector = csv.DictReader(f)
    filas = list(lector)

random.seed(42)
random.shuffle(filas)

textos = [fila['instruction'] for fila in filas]
categorias = [fila['category'] for fila in filas]

conteo_global = Counter()
for texto in textos:
    conteo_global.update(preprocesar(texto))

vocabulario = sorted([p for p, c in conteo_global.items() if c > 2])

#ENTRENAR CON EL DATASET COMPLETO

def entrenar(textos, categorias, vocabulario):
    clases = set(categorias)
    total = len(textos)
    conteo_clases = Counter(categorias)
    prob_prior = {c: math.log(conteo_clases[c] / total) for c in clases}
    palabras_por_clase = {c: Counter() for c in clases}
    for texto, cat in zip(textos, categorias):
        palabras_por_clase[cat].update(preprocesar(texto))
    V = len(vocabulario)
    prob_palabras = {}
    for c in clases:
        total_palabras = sum(palabras_por_clase[c].values())
        prob_palabras[c] = {}
        for palabra in vocabulario:
            conteo = palabras_por_clase[c].get(palabra, 0)
            prob_palabras[c][palabra] = math.log((conteo + 1) / (total_palabras + V))
    return prob_prior, prob_palabras

print("Entrenando modelo con todo el dataset...")
prob_prior, prob_palabras = entrenar(textos, categorias, vocabulario)
print("Modelo entrenado!")

#GUARDAR MODELO

modelo = {
    "prob_prior": prob_prior,
    "prob_palabras": prob_palabras,
    "vocabulario": vocabulario
}

with open("../modelo/modelo_naive_bayes.pkl", "wb") as f:
    pickle.dump(modelo, f)

print("Modelo guardado en: modelo/modelo_naive_bayes.pkl")

#VERIFICAR SI CARGA
with open("../modelo/modelo_naive_bayes.pkl", "rb") as f:
    modelo_cargado = pickle.load(f)

print("Modelo cargado correctamente")
print(f"Clases disponibles: {sorted(modelo_cargado['prob_prior'].keys())}")