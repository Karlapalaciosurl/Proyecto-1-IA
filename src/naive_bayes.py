import csv
import re
import math
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

#Preprocesar, cargar el archivo y vocabulario
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

textos = [fila['instruction'] for fila in filas]
categorias = [fila['category'] for fila in filas]

conteo_global = Counter()
for texto in textos:
    conteo_global.update(preprocesar(texto))

vocabulario = sorted([p for p, c in conteo_global.items() if c > 2])
vocab_index = {p: i for i, p in enumerate(vocabulario)}

print(f"Vocabulario: {len(vocabulario)} palabras")

#-------------------------------------------------------------------------------------
#NAIVE BAYES 

def entrenar(textos, categorias, vocabulario):
    clases = set(categorias)
    total = len(textos)
    
    # Probabilidades a priori
    conteo_clases = Counter(categorias)
    prob_prior = {c: math.log(conteo_clases[c] / total) for c in clases}
    
    # Contar palabras por clase
    palabras_por_clase = {c: Counter() for c in clases}
    for texto, cat in zip(textos, categorias):
        tokens = preprocesar(texto)
        palabras_por_clase[cat].update(tokens)
    
    # Laplace Smoothing
    V = len(vocabulario)
    prob_palabras = {}
    for c in clases:
        total_palabras = sum(palabras_por_clase[c].values())
        prob_palabras[c] = {}
        for palabra in vocabulario:
            conteo = palabras_por_clase[c].get(palabra, 0)
            # Laplace Smoothing: (conteo + 1) / (total + V)
            prob_palabras[c][palabra] = math.log((conteo + 1) / (total_palabras + V))
    
    return prob_prior, prob_palabras

#REALIZAR PREDICCIONES

def predecir(texto, prob_prior, prob_palabras, vocabulario):
    tokens = preprocesar(texto)
    clases = prob_prior.keys()
    scores = {}
    
    for c in clases:
        # Empezar con el prior
        score = prob_prior[c]
        # Sumar log probabilidades de cada palabra
        for token in tokens:
            if token in vocabulario:
                score += prob_palabras[c][token]
        scores[c] = score
    
    return max(scores, key=scores.get)

#PRUEBAS: 

prob_prior, prob_palabras = entrenar(textos, categorias, vocabulario)
ejemplos = [
    "I want to cancel my order",
    "I need help with my invoice",
    "where is my package",
    "I want a refund please",
    "I need and account please"
]

print("\nPREDICCIONES DE PRUEBA")
for e in ejemplos:
    pred = predecir(e, prob_prior, prob_palabras, vocabulario)
    print(f"'{e}' → {pred}")