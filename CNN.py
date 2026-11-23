import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. CARGAR Y PREPARAR LOS DATOS
# Usamos Fashion-MNIST: 70,000 imágenes en blanco y negro de 28x28 píxeles
(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.fashion_mnist.load_data()
)

nombres_clases = [
    'Camiseta',
    'Pantalón',
    'Suéter',
    'Vestido',
    'Abrigo',
    'Sandalia',
    'Camisa',
    'Tenis',
    'Bolso',
    'Botín',
]

# Normalizar píxeles de [0, 255] a [0, 1] y agregar dimensión de canal (1 = escala de grises)
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# 2. CONSTRUIR LA ARQUITECTURA DE LA CNN
modelo = models.Sequential([
    # Primera capa convolucional: 32 filtros de 3x3
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    # Segunda capa convolucional: 64 filtros de 3x3 para patrones más complejos
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    # Clasificador final
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax'),  # 10 salidas (una por cada prenda)
])

# 3. COMPILAR Y ENTRENAR
modelo.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

print('Entrenando la red convolucional...')
modelo.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# 4. EVALUAR CON DATOS NUEVOS
perdida, precision = modelo.evaluate(x_test, y_test, verbose=0)
print(f'\nPrecisión en imágenes de prueba: {precision * 100:.2f}%')

# 5. HACER UNA PREDICCIÓN DE EJEMPLO
imagen_ejemplo = x_test[0:1]  # Tomamos la primera imagen del set de prueba
prediccion = modelo.predict(imagen_ejemplo, verbose=0)
clase_predicha = np.argmax(prediccion)
clase_real = y_test[0]

print(f'Predicción del modelo: {nombres_clases[clase_predicha]}')
print(f'Etiqueta real:         {nombres_clases[clase_real]}')