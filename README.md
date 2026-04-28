# Proyecto 1 - Clasificación de Solicitudes a Mesa de Ayuda
**Universidad Rafael Landívar - Inteligencia Artificial**

Clasificador de tickets de soporte al cliente usando Naïve Bayes implementado desde cero.

## ¿Qué hace?
Clasifica solicitudes de clientes en 11 categorías:
ACCOUNT, CANCEL, CONTACT, DELIVERY, FEEDBACK, INVOICE, ORDER, PAYMENT, REFUND, SHIPPING, SUBSCRIPTION

## Tecnologías usadas
- Python
- NLTK (tokenización y stopwords)
- Flask (servidor web)
- HTML, CSS y JavaScript

## Instalación
1. Clonar el repositorio
2. Instalar las dependencias: pip install nltk flask

## ¿Cómo correr el proyecto?
**Entrenar y guardar el modelo:**
cd src
python guardar_modelo.py

**Correr la página web:**
cd web
python app.py
Luego abrir el navegador en: http://localhost:5000

## Estructura del proyecto
├── modelo/          --modelo entrenado (.pkl)
├── src/             --scripts de Python
│   ├── explorar.py
│   ├── limpieza_datos.py
│   ├── bag_of_words.py
│   ├── naive_bayes.py
│   ├── kfolds.py
│   ├── metricas.py
│   └── guardar_modelo.py
├── web/             -- página web
│   ├── app.py
│   └── templates/
│       └── index.html
└── README.md
