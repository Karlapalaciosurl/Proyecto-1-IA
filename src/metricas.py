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


# DIVIDIR DATOS (80% entrenamiento, 20% prueba)

corte = int(len(textos) * 0.8)
textos_train = textos[:corte]
cats_train = categorias[:corte]
textos_prueba = textos[corte:]
cats_prueba = categorias[corte:]

# Entrenar
prob_prior, prob_palabras = entrenar(textos_train, cats_train, vocabulario)

# Predecir
predicciones = [predecir(t, prob_prior, prob_palabras, vocabulario) for t in textos_prueba]


# MÉTRICAS POR CLASE


clases = sorted(set(categorias))


print(f"{'Clase':<15} {'Precisión':>10} {'Recall':>10} {'F1-Score':>10}")


f1_scores = []

for c in clases:
    tp = sum(1 for r, p in zip(cats_prueba, predicciones) if r == c and p == c)
    fp = sum(1 for r, p in zip(cats_prueba, predicciones) if r != c and p == c)
    fn = sum(1 for r, p in zip(cats_prueba, predicciones) if r == c and p != c)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    f1_scores.append(f1)

    print(f"{c:<15} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f}")

accuracy = sum(1 for r, p in zip(cats_prueba, predicciones) if r == p) / len(cats_prueba)
macro_f1 = sum(f1_scores) / len(f1_scores)
print(f"{'Exactitud global:':<30} {accuracy:.4f}")
print(f"{'Macro F1:':<30} {macro_f1:.4f}")

# MATRIZ DE CONFUSION

print("\nMATRIZ DE CONFUSION\n")
print(f"{'':>12}", end="")
for c in clases:
    print(f"{c:>12}", end="")
print()

for real in clases:
    print(f"{real:>12}", end="")
    for pred in clases:
        count = sum(1 for r, p in zip(cats_prueba, predicciones) if r == real and p == pred)
        print(f"{count:>12}", end="")
    print()