# Laboratorio 3 Introducción a la Inteligencia artificial
# 🧬 Algoritmo Genético para Optimización de una Función Objetivo

Este proyecto implementa un **Algoritmo Genético (AG)** en Python para encontrar el **máximo global** de una función no lineal.  
El AG utiliza operadores clásicos de selección, cruce y mutación, y visualiza el proceso de convergencia y el resultado final.

---

## 📌 Descripción General

La función objetivo que se desea maximizar es:

\[
f(x) = x \cdot \sin(10 \pi x) + 2
\]

El algoritmo busca el valor de `x` dentro del rango \([-1, 2]\) que maximiza esta función.

---

## 🧠 Algoritmo Utilizado

El flujo del algoritmo genético es el siguiente:

1. **Inicializar población** → Se generan individuos aleatorios dentro de los límites definidos.  
2. **Evaluar fitness** → Se calcula la función objetivo para cada individuo.  
3. **Selección por torneo** → Se eligen los mejores candidatos de forma competitiva.  
4. **Cruce (reproducción)** → Se combinan padres para crear nuevos individuos.  
5. **Mutación** → Se introduce aleatoriedad controlada para mantener diversidad.  
6. **Reemplazo** → La nueva generación sustituye a la anterior.  
7. **Repetición** → Se itera durante un número fijo de generaciones.  
8. **Salida** → Se obtiene la mejor solución encontrada.

---

## 🧰 Tecnologías Usadas

- 🐍 **Python 3**
- 📊 **NumPy** — para manejo eficiente de arreglos numéricos.
- 📈 **Matplotlib** — para graficar convergencia y función objetivo.

---

## 📂 Estructura del Código

```python
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
⚙️ Parámetros Principales
Parámetro	Valor por defecto	Descripción
bounds	(-1, 2)	Rango de búsqueda para la solución.
pop_size	50	Número de individuos en la población.
generations	100	Número de generaciones de evolución.
mutation_rate	0.1	Probabilidad de mutación de un individuo.
rate (crossover)	0.8	Probabilidad de que ocurra cruce entre dos padres.

📊 Resultados Visuales
Durante la ejecución se generan dos gráficos:

📈 Convergencia del fitness máximo a lo largo de las generaciones.

🟥 Curva de la función objetivo con el máximo encontrado resaltado.

Ejemplo de salida por consola:

yaml
Copiar código
Mejor solución encontrada: 1.850234 con fitness: 2.98217
