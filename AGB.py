import random

# --- PARÁMETROS DEL ALGORITMO ---
LONGITUD_CROMOSOMA = 20  # Cantidad de bits por individuo
TAMANO_POBLACION = 30  # Cantidad de individuos por generación
PROB_CRUCE = 0.8  # 80% de probabilidad de que dos padres se crucen
PROB_MUTACION = 0.05  # 5% de probabilidad de que un bit mute
GENERACIONES = 50  # Límite de generaciones


def crear_individuo(longitud):
  """Genera un cromosoma binario aleatorio (ej. [0, 1, 1, 0...])."""
  return [random.randint(0, 1) for _ in range(longitud)]


def calcular_fitness(individuo):
  """Función de aptitud: cuenta cuántos '1' tiene el individuo."""
  return sum(individuo)


def seleccion_torneo(poblacion, k=3):
  """Elige k individuos al azar y devuelve el que tenga mejor fitness."""
  competidores = random.sample(poblacion, k)
  return max(competidores, key=calcular_fitness)


def cruzar(padre1, padre2):
  """Cruce de un punto: combina la primera mitad de un padre con la segunda del otro."""
  if random.random() < PROB_CRUCE:
    punto = random.randint(1, len(padre1) - 1)
    hijo1 = padre1[:punto] + padre2[punto:]
    hijo2 = padre2[:punto] + padre1[punto:]
    return hijo1, hijo2
  return padre1[:], padre2[:]


def mutar(individuo):
  """Invierte bits (0 -> 1 o 1 -> 0) según la probabilidad de mutación."""
  for i in range(len(individuo)):
    if random.random() < PROB_MUTACION:
      individuo[i] = 1 - individuo[i]
  return individuo


# --- BUCLE EVOLUTIVO PRINCIPAL ---

# 1. Crear población inicial
poblacion = [
    crear_individuo(LONGITUD_CROMOSOMA) for _ in range(TAMANO_POBLACION)
]

for gen in range(GENERACIONES):
  # Ordenar población de mejor a peor para monitorear el progreso
  poblacion.sort(key=calcular_fitness, reverse=True)
  mejor_actual = poblacion[0]
  mejor_fitness = calcular_fitness(mejor_actual)

  print(
      f"Gen {gen:02d} | Mejor fitness: {mejor_fitness}/{LONGITUD_CROMOSOMA} |"
      f" {''.join(map(str, mejor_actual))}"
  )

  # Detener si ya encontramos la solución óptima
  if mejor_fitness == LONGITUD_CROMOSOMA:
    print("\n¡Solución óptima alcanzada!")
    break

  # Elitismo: conservar al mejor individuo intacto para la siguiente generación
  nueva_poblacion = [mejor_actual[:]]

  # Llenar el resto de la nueva generación
  while len(nueva_poblacion) < TAMANO_POBLACION:
    padre1 = seleccion_torneo(poblacion)
    padre2 = seleccion_torneo(poblacion)

    hijo1, hijo2 = cruzar(padre1, padre2)

    nueva_poblacion.append(mutar(hijo1))
    if len(nueva_poblacion) < TAMANO_POBLACION:
      nueva_poblacion.append(mutar(hijo2))

  poblacion = nueva_poblacion