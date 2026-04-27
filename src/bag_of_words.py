import csv
import re
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

# CARGAR DATASET
archivo = "../Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"

with open(archivo, encoding="utf-8") as f:
    lector = csv.DictReader(f)
    filas = list(lector)

# --------------------------------------------------------------
#VOCABULARIO

vocabulario = set()

for fila in filas:
    tokens = preprocesar(fila['instruction'])
    vocabulario.update(tokens)

#vocabulario = sorted(vocabulario)
# Contar cuántas veces aparece cada palabra en todo el dataset
conteo_global = Counter()
for fila in filas:
    tokens = preprocesar(fila['instruction'])
    conteo_global.update(tokens)

# Filtro para conservar solo las palabras que aparecen más de 2 veces
vocabulario = sorted([palabra for palabra, conteo in conteo_global.items() if conteo > 2])

print(f"Tamaño del vocabulario: {len(vocabulario)}")
print(f"Primeras 20 palabras: {vocabulario[:20]}")

# --------------------------------------------------------------
# REPRESENTAR  DOCUMENTO COMO VECTOR DE FRECUENCIA DE PALABRAS 

def texto_a_vector(texto, vocab):
    tokens = preprocesar(texto)
    conteo = Counter(tokens)
    vector = [conteo.get(palabra, 0) for palabra in vocab]
    return vector

# Probar con la primera fila
ejemplo = filas[0]['instruction']
vector = texto_a_vector(ejemplo, vocabulario)

print(f"\nEjemplo de texto: {ejemplo}")
print(f"Palabras con frecuencia > 0: {sum(1 for v in vector if v > 0)}")
print(f"Tamaño del vector: {len(vector)}")