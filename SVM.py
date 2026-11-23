from sklearn import svm
#caracteristicas de entrenamiento
x = [
    [4, 2],
    [8, 2],
    [8, 4],
    [12, 4],
    [16, 6],
    [16, 8]
]
#respuestas de entrenamiento donde 0 es no adecuada y 1 es adecuada
y = [0, 0, 1, 1, 1, 1]
#crea el modelo
modelo = svm.SVC(kernel='linear')
#entrena el mdelo
modelo.fit(x, y)
#registra la nueva pc
nueva_pc = [[5, 2]]
#hace la prediccion
prediccion = modelo.predict(nueva_pc)

if prediccion[0] == 1:
    print("La nueva pc es adecuada")
else:
    print("La nueva pc no es adecuada")
print("Clase predicha", prediccion[0])

