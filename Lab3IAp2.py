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
