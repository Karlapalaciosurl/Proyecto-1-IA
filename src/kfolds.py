import csv
import re
import math
import random
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

# NAIVE BAYES
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

def predecir(texto, prob_prior, prob_palabras, vocabulario):
    tokens = preprocesar(texto)
    scores = {}
    for c in prob_prior:
        score = prob_prior[c]
        for token in tokens:
            if token in vocabulario:
                score += prob_palabras[c][token]
        scores[c] = score
    return max(scores, key=scores.get)

# ------------------------------------------------------------------------------------
# K-FOLDS(K=5)

K = 5
n = len(textos)
tamano_fold = n // K

print(f"Total de datos: {n}")
print(f"Tamaño de cada fold: {tamano_fold}")
print(f"Corriendo {K}-Folds...\n")

accuracy_por_fold = []
clases = sorted(set(categorias))

for k in range(K):
    # Separar en entrenamiento y prueba
    inicio = k * tamano_fold
    fin = inicio + tamano_fold

    textos_prueba = textos[inicio:fin]
    cats_prueba = categorias[inicio:fin]
    textos_train = textos[:inicio] + textos[fin:]
    cats_train = categorias[:inicio] + categorias[fin:]

    # Entrenar
    prob_prior, prob_palabras = entrenar(textos_train, cats_train, vocabulario)

    # Evaluar
    correctos = 0
    for texto, cat_real in zip(textos_prueba, cats_prueba):
        pred = predecir(texto, prob_prior, prob_palabras, vocabulario)
        if pred == cat_real:
            correctos += 1

    acc = correctos / len(textos_prueba)
    accuracy_por_fold.append(acc)
    print(f"Fold {k+1}: Exactitud = {acc:.4f}")

print(f"\nExactitud promedio: {sum(accuracy_por_fold)/K:.4f}")