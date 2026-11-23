import math
import random


def distancia_euclidiana(punto1, punto2):
  """Calcula la distancia en línea recta entre dos puntos."""
  return math.sqrt(sum((a - b) ** 2 for a, b in zip(punto1, punto2)))


def calcular_promedio(puntos):
  """Calcula el centroide (promedio por cada coordenada) de un grupo de puntos."""
  num_dimensiones = len(puntos[0])
  centroide = []
  for dim in range(num_dimensiones):
    promedio_dim = sum(p[dim] for p in puntos) / len(puntos)
    centroide.append(promedio_dim)
  return centroide


def kmeans(datos, k=2, max_iteraciones=100):
  """Agrupa los datos en k clústeres y devuelve los centroides y las asignaciones."""
  # 1. Elegir k puntos aleatorios como centroides iniciales (sin repetir)
  centroides = random.sample(datos, k)

  for iteracion in range(max_iteraciones):
    # Crear listas vacías para guardar los puntos de cada grupo
    grupos = [[] for _ in range(k)]

    # 2. Asignar cada punto al centroide más cercano
    for punto in datos:
      distancias = [distancia_euclidiana(punto, c) for c in centroides]
      grupo_mas_cercano = distancias.index(min(distancias))
      grupos[grupo_mas_cercano].append(punto)

    # Guardar los centroides anteriores para ver si hay cambios
    nuevos_centroides = []

    # 3. Recalcular el centro de cada grupo
    for i in range(k):
      if grupos[i]:  # Evitar división por cero si un grupo queda vacío
        nuevos_centroides.append(calcular_promedio(grupos[i]))
      else:
        nuevos_centroides.append(random.choice(datos))

    # 4. Verificar si los centroides dejaron de moverse
    if nuevos_centroides == centroides:
      print(f"Convergió en la iteración {iteracion + 1}")
      break

    centroides = nuevos_centroides

  return centroides, grupos


# --- EJEMPLO DE USO ---

# Datos sin etiquetas: [Horas de estudio a la semana, Calificación]
estudiantes = [
    [1.0, 5.0],
    [1.5, 5.5],
    [2.0, 6.0],  # Grupo bajo rendimiento
    [8.0, 9.0],
    [8.5, 9.5],
    [9.0, 9.8],  # Grupo alto rendimiento
    [4.5, 7.2],
    [5.0, 7.5],
    [5.5, 7.8],  # Grupo rendimiento medio
]

centroides_finales, grupos_formados = kmeans(estudiantes, k=3)

print("\n--- RESULTADOS ---")
for i, (centro, grupo) in enumerate(zip(centroides_finales, grupos_formados)):
  centro_redondeado = [round(coord, 2) for coord in centro]
  print(f"Grupo {i + 1} | Centroide: {centro_redondeado}")
  print(f"   Puntos: {grupo}\n")