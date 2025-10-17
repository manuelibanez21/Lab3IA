# 🚚 Algoritmo Genético para Resolver el Problema del Viajero (TSP)

Este proyecto implementa un **Algoritmo Genético (AG)** para resolver una versión del **Problema del Viajero (TSP)**.  
El objetivo es encontrar la ruta más corta que visita todas las ciudades exactamente una vez y regresa a la ciudad inicial.

---

## 📌 Descripción del Problema

El **Problema del Viajero (TSP)** consiste en encontrar el **camino mínimo** para recorrer `n` ciudades pasando una sola vez por cada una y volviendo al punto de partida.  
Es un problema clásico de optimización combinatoria y NP-difícil, por lo que no existe una solución exacta eficiente para grandes cantidades de ciudades.

Este código implementa un algoritmo genético que aproxima una **ruta óptima** a través de técnicas evolutivas.

---

## 🧠 Flujo del Algoritmo Genético TSP

1. 🏙️ **Generar ciudades:** Se crean posiciones aleatorias para las ciudades en un plano 2D.  
2. 📏 **Calcular distancias:** Se obtiene la distancia total de cada ruta propuesta.  
3. 🧬 **Inicializar población:** Se generan rutas aleatorias iniciales (permutaciones de ciudades).  
4. 🏁 **Evaluar fitness:** Se calcula el fitness como el inverso de la distancia total.  
5. ✨ **Selección:** Se eligen individuos proporcionalmente a su fitness.  
6. 🔀 **Cruce (OX1):** Se cruzan dos rutas para producir nuevas soluciones.  
7. ♻️ **Mutación:** Se intercambian ciudades aleatoriamente con cierta probabilidad.  
8. 🧭 **Repetir:** Se itera este proceso por un número determinado de generaciones.  
9. 🏆 **Salida:** Se obtiene la mejor ruta encontrada y se grafica.

---

## 🧰 Tecnologías Usadas

- 🐍 **Python 3**
- 📊 **NumPy** — para operaciones numéricas y manejo de arreglos.
- 📈 **Matplotlib** — para graficar convergencia y ruta final.

---

## 📂 Código Fuente

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Crear ciudades ---
np.random.seed(42)
cities = np.random.rand(10, 2) * 100

# --- Distancia total de una ruta ---
def route_distance(route):
    return np.sum(np.linalg.norm(np.diff(cities[route], axis=0), axis=1)) + \
           np.linalg.norm(cities[route[0]] - cities[route[-1]])

# --- Fitness ---
def fitness(route):
    return 1 / route_distance(route)

# --- Inicializar población ---
def init_population(pop_size, n_cities):
    return [np.random.permutation(n_cities) for _ in range(pop_size)]

# --- Selección ---
def selection(pop, fitness_values):
    idx = np.random.choice(len(pop), size=len(pop), p=fitness_values/fitness_values.sum())
    return [pop[i] for i in idx]

# --- Cruce (OX1) ---
def crossover(parent1, parent2):
    a, b = sorted(np.random.choice(len(parent1), 2, replace=False))
    child = [-1]*len(parent1)
    child[a:b] = parent1[a:b]
    pos = b
    for city in parent2:
        if city not in child:
            if pos == len(parent1): pos = 0
            child[pos] = city
            pos += 1
    return child

# --- Mutación (swap) ---
def mutation(route, rate=0.1):
    if np.random.rand() < rate:
        i, j = np.random.choice(len(route), 2, replace=False)
        route[i], route[j] = route[j], route[i]
    return route

# --- Algoritmo Genético TSP ---
def genetic_algorithm_tsp(pop_size=100, generations=200, mutation_rate=0.2):
    population = init_population(pop_size, len(cities))
    best_route, best_dist = None, float("inf")
    history = []

    for g in range(generations):
        fitness_values = np.array([fitness(route) for route in population])
        best_idx = np.argmax(fitness_values)
        dist = route_distance(population[best_idx])
        history.append(dist)
        if dist < best_dist:
            best_route, best_dist = population[best_idx], dist

        selected = selection(population, fitness_values)
        children = []
        for i in range(0, len(selected), 2):
            p1, p2 = selected[i], selected[(i+1)%len(selected)]
            c1, c2 = crossover(p1, p2), crossover(p2, p1)
            children.append(mutation(c1, mutation_rate))
            children.append(mutation(c2, mutation_rate))
        population = children

    # Gráfica convergencia
    plt.plot(history)
    plt.xlabel("Generaciones")
    plt.ylabel("Distancia mínima")
    plt.title("Convergencia - TSP")
    plt.show()

    return best_route, best_dist

# --- Ejecución ---
best_route, best_dist = genetic_algorithm_tsp()
print("Mejor ruta encontrada:", best_route, "con distancia:", best_dist)

# --- Gráfica de la mejor ruta ---
route_coords = cities[best_route]
route_coords = np.vstack([route_coords, route_coords[0]])  # cerrar el ciclo

plt.figure(figsize=(6,6))
plt.scatter(cities[:,0], cities[:,1], c="blue", label="Ciudades")
plt.plot(route_coords[:,0], route_coords[:,1], c="red", lw=2, label="Ruta óptima")
for i, (x, y) in enumerate(cities):
    plt.text(x+1, y+1, str(i), fontsize=9)
plt.xlabel("X")
plt.ylabel("Y")
plt.title(f"Mejor ruta encontrada - Distancia: {best_dist:.2f}")
plt.legend()
plt.show()
🧾 Explicación Paso a Paso
🟡 Crear ciudades
python
Copiar código
cities = np.random.rand(10, 2) * 100
Se generan 10 coordenadas aleatorias en el plano.

Cada ciudad tiene una posición (x, y).

📏 Calcular distancia de una ruta
python
Copiar código
def route_distance(route):
    ...
Calcula la distancia total recorrida al visitar todas las ciudades en el orden dado por route.

Incluye la distancia de regreso a la ciudad inicial (circuito cerrado).

🟢 Fitness
python
Copiar código
def fitness(route):
    return 1 / route_distance(route)
El fitness es inversamente proporcional a la distancia:

Rutas más cortas → mayor fitness.

Rutas más largas → menor fitness.

🧬 Inicialización de población
python
Copiar código
def init_population(pop_size, n_cities):
    return [np.random.permutation(n_cities) for _ in range(pop_size)]
Se crean rutas iniciales aleatorias (permutaciones de las ciudades).

🔵 Selección proporcional al fitness
python
Copiar código
def selection(pop, fitness_values):
    idx = np.random.choice(len(pop), size=len(pop), p=fitness_values/fitness_values.sum())
    return [pop[i] for i in idx]
Las rutas con mayor fitness tienen mayor probabilidad de ser seleccionadas como padres.

🟠 Cruce OX1 (Order Crossover 1)
python
Copiar código
def crossover(parent1, parent2):
    ...
Se copia un segmento de parent1 y se completa el hijo con el orden de ciudades de parent2 sin duplicar.

Mantiene validez de la ruta (todas las ciudades aparecen una vez).

🔴 Mutación (swap)
python
Copiar código
def mutation(route, rate=0.1):
    ...
Con probabilidad rate, intercambia dos ciudades de la ruta.

Esto introduce diversidad genética y evita el estancamiento.

🔁 Bucle evolutivo
Para cada generación:

Se evalúan los fitness.

Se seleccionan padres.

Se cruzan y mutan para generar nueva población.

Se guarda la mejor ruta encontrada hasta el momento.

📈 Gráfica de convergencia
Muestra cómo la distancia mínima encontrada mejora a lo largo de las generaciones.

🏆 Resultado final
Se imprime la mejor ruta y su distancia.

Se muestra un gráfico con la trayectoria óptima encontrada.

Ejemplo de salida en consola:

yaml
Copiar código
Mejor ruta encontrada: [0 7 2 9 3 1 4 8 5 6] con distancia: 267.3548
⚙️ Parámetros Principales
Parámetro	Valor por defecto	Descripción
pop_size	100	Tamaño de la población.
generations	200	Número de generaciones del algoritmo.
mutation_rate	0.2	Probabilidad de mutación de una ruta.
n_cities	10	Número de ciudades en el mapa.

📊 Resultados Visuales
📈 Gráfico de convergencia: distancia mínima por generación.

🗺️ Gráfico de ruta óptima: muestra el orden de las ciudades y la distancia final.
