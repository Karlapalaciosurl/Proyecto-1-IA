import csv
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# FUNCIONES DE PREPROCESAMIENTO
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def limpiar_texto(texto):
    # Quitar placeholders como {{Order Number}}
    texto = re.sub(r'\{\{.*?\}\}', '', texto)
    #Pasar a minúsculas
    texto = texto.lower()
    # Quitar caracteres especiales y números
    texto = re.sub(r'[^a-z\s]', '', texto)
    #Quitar espacios extra
    texto = texto.strip()
    return texto

def tokenizar(texto):
    return texto.split()

def quitar_stopwords(tokens):
    return [t for t in tokens if t not in stop_words]

def lematizar(tokens):
    return [stemmer.stem(t) for t in tokens]

def preprocesar(texto):
    texto = limpiar_texto(texto)
    tokens = tokenizar(texto)
    tokens = quitar_stopwords(tokens)
    tokens = lematizar(tokens)
    return tokens


# PROBAR DATASET

archivo = "../Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"

with open(archivo, encoding="utf-8") as f:
    lector = csv.DictReader(f)
    filas = list(lector)

for fila in filas[:3]:
    texto_original = fila['instruction']
    tokens = preprocesar(texto_original)
    print(f"Original : {texto_original}")
    print(f"Procesado: {tokens}")
    print(f"Categoría: {fila['category']}")
    print()