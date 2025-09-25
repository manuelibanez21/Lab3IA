import numpy as np
import matplotlib.pyplot as plt

# --- Función objetivo ---
def fitness_function(x):
    return x * np.sin(10 * np.pi * x) + 2.0

# --- Inicializar población ---
def init_population(size, bounds):
    return np.random.uniform(bounds[0], bounds[1], size)

# --- Selección por torneo ---
def selection(pop, fitness, k=3):
    selected = []
    for _ in range(len(pop)):
        aspirants_idx = np.random.randint(0, len(pop), k)
        best = aspirants_idx[np.argmax(fitness[aspirants_idx])]
        selected.append(pop[best])
    return np.array(selected)

# --- Cruce uniforme ---
def crossover(parent1, parent2, rate=0.8):
    if np.random.rand() < rate:
        alpha = np.random.rand()
        return alpha * parent1 + (1 - alpha) * parent2
    return parent1

# --- Mutación ---
def mutation(child, bounds, rate=0.1):
    if np.random.rand() < rate:
        child += np.random.uniform(-0.1, 0.1)
        child = np.clip(child, bounds[0], bounds[1])
    return child

# --- Algoritmo Genético ---
def genetic_algorithm(bounds=(-1, 2), pop_size=50, generations=100, mutation_rate=0.1):
    population = init_population(pop_size, bounds)
    best_fitness_evolution = []

    for g in range(generations):
        fitness = fitness_function(population)
        best_fitness_evolution.append(np.max(fitness))

        selected = selection(population, fitness)
        children = []
        for i in range(0, len(selected), 2):
            p1, p2 = selected[i], selected[(i+1) % len(selected)]
            child1 = crossover(p1, p2)
            child2 = crossover(p2, p1)
            child1 = mutation(child1, bounds, mutation_rate)
            child2 = mutation(child2, bounds, mutation_rate)
            children.append(child1)
            children.append(child2)
        population = np.array(children)

    # Gráfica de convergencia
    plt.plot(best_fitness_evolution)
    plt.xlabel("Generaciones")
    plt.ylabel("Fitness máximo")
    plt.title("Convergencia - Función Objetivo")
    plt.show()

    best_solution = population[np.argmax(fitness_function(population))]
    return best_solution, np.max(fitness_function(population))

# --- Ejecución ---
best_x, best_y = genetic_algorithm()
print("Mejor solución encontrada:", best_x, "con fitness:", best_y)

# --- Gráfica de la función y el máximo ---
x_vals = np.linspace(-1, 2, 400)
y_vals = fitness_function(x_vals)

plt.plot(x_vals, y_vals, label="Función objetivo")
plt.scatter(best_x, best_y, color="red", label="Máximo encontrado", zorder=5)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Máximo global encontrado")
plt.legend()
plt.show()
