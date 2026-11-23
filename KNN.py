from sklearn.neighbors import KNeighborsClassifier

# Datos de entrenamiento
X = [
    [4, 2],
    [8, 2],
    [8, 4],
    [12, 4],
    [16, 6],
    [16, 8]
]

# Clases
y = [0, 0, 1, 1, 1, 1]

# Crear el modelo KNN
modelo = KNeighborsClassifier(n_neighbors=3)

# Entrenar el modelo
modelo.fit(X, y)

# Nueva PC
nueva_pc = [[12, 5]]

# Realizar predicción
prediccion = modelo.predict(nueva_pc)

# Mostrar resultado
if prediccion[0] == 1:
    print("La PC pertenece a la clase: Adecuada")
else:
    print("La PC pertenece a la clase: No adecuada")

    