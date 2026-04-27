import csv

archivo = "../Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"

with open(archivo, encoding="utf-8") as f:
    lector = csv.reader(f)
    encabezados = next(lector)
    filas = list(lector)

print("COLUMNAS")
print(encabezados)

print(f"\nTOTAL DE FILAS")
print(len(filas))

print("\nPRIMERAS 3 FILAS")
for fila in filas[:3]:
    print(fila)

print("\nCATEGORÍAS ÚNICAS")
idx = encabezados.index("category")
categorias = set(fila[idx] for fila in filas)
print(sorted(categorias))