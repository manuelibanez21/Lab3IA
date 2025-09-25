import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

class ScheduleOptimizer:
    def __init__(self):
        # Profesores y sus disponibilidades
        self.teachers = {
            "ProfA": ["Lun-9:00", "Mar-9:00", "Mie-9:00"],       # solo en la mañana
            "ProfB": ["Lun-11:00", "Mar-11:00", "Mie-11:00"],    # solo en la tarde
            "ProfC": ["Lun-9:00", "Lun-11:00", "Mar-9:00"],      # mixto
            "ProfD": ["Mar-11:00", "Mie-9:00", "Mie-11:00"],     # mixto
        }

        self.subjects = ["Matemáticas", "Ciencias", "Historia", "Arte"]
        self.groups = ["Grupo1", "Grupo2", "Grupo3"]
        self.time_slots = ["Lun-9:00", "Lun-11:00", "Mar-9:00", "Mar-11:00", "Mie-9:00", "Mie-11:00"]

        # Preferencias de materias (ejemplo: Matemáticas en la mañana)
        self.preferences = {
            "Matemáticas": ["Lun-9:00", "Mar-9:00", "Mie-9:00"],
            "Ciencias": ["Lun-11:00", "Mar-11:00"],
            "Historia": self.time_slots,  # sin restricción
            "Arte": self.time_slots       # sin restricción
        }

    def create_schedule(self):
        """Crea un horario aleatorio válido con disponibilidad de profes"""
        schedule = []
        for time in self.time_slots:
            for group in self.groups:
                teacher = random.choice(list(self.teachers.keys()))
                subject = random.choice(self.subjects)
                schedule.append((time, group, teacher, subject))
        return schedule

    def evaluate_schedule(self, schedule):
        """Calcula el fitness basado en restricciones"""
        score = 0

        for i in range(len(schedule)):
            t1, g1, p1, s1 = schedule[i]
            # Penalizar si profesor no está disponible
            if t1 not in self.teachers[p1]:
                score -= 3

            # Penalizar si materia no respeta preferencias
            if t1 not in self.preferences[s1]:
                score -= 2

            # Revisar choques
            for j in range(i+1, len(schedule)):
                t2, g2, p2, s2 = schedule[j]
                if g1 == g2 and t1 == t2:
                    score -= 5  # choque grupo
                if p1 == p2 and t1 == t2:
                    score -= 5  # choque profe

        # Recompensa si no hay choques
        score += 10 * len(set([g for _, g, _, _ in schedule]))
        return score

    def mutation(self, schedule):
        """Cambia aleatoriamente el horario de una clase"""
        i = random.randint(0, len(schedule)-1)
        time, group, teacher, subject = schedule[i]
        if random.random() < 0.5:
            # cambiar profesor
            teacher = random.choice(list(self.teachers.keys()))
        else:
            # cambiar materia
            subject = random.choice(self.subjects)
        schedule[i] = (time, group, teacher, subject)
        return schedule

    def crossover(self, s1, s2):
        """Cruce de dos horarios"""
        point = random.randint(0, len(s1)-1)
        return s1[:point] + s2[point:]

    def genetic_algorithm(self, pop_size=50, generations=100, mutation_rate=0.2):
        """Algoritmo genético para optimizar horarios"""
        population = [self.create_schedule() for _ in range(pop_size)]
        best, best_fit = None, -9999
        history = []

        for g in range(generations):
            fits = np.array([self.evaluate_schedule(h) for h in population])
            best_idx = np.argmax(fits)
            if fits[best_idx] > best_fit:
                best_fit = fits[best_idx]
                best = population[best_idx]
            history.append(best_fit)

            selected = [population[i] for i in np.random.choice(len(population), len(population))]
            children = []
            for i in range(0, len(selected), 2):
                c1 = self.crossover(selected[i], selected[(i+1)%len(selected)])
                c2 = self.crossover(selected[(i+1)%len(selected)], selected[i])
                if random.random() < mutation_rate:
                    c1 = self.mutation(c1)
                if random.random() < mutation_rate:
                    c2 = self.mutation(c2)
                children.append(c1)
                children.append(c2)
            population = children

        # Gráfica convergencia
        plt.plot(history)
        plt.xlabel("Generaciones")
        plt.ylabel("Fitness")
        plt.title("Convergencia - Optimización de Horarios")
        plt.show()

        return best, best_fit


# --- Ejecución ---
optimizer = ScheduleOptimizer()
best_horario, best_score = optimizer.genetic_algorithm()
print("Mejor horario encontrado (fitness:", best_score, ")")
for h in best_horario:
    print(h)

# --- Representar en tabla ---
horario_df = pd.DataFrame(index=optimizer.groups, columns=optimizer.time_slots)
for time, group, teacher, subject in best_horario:
    if pd.isna(horario_df.loc[group, time]):
        horario_df.loc[group, time] = f"{subject}-{teacher}"
    else:
        horario_df.loc[group, time] += f" / {subject}-{teacher}"

print("\nHorario en tabla:")
print(horario_df)

# --- Gráfica tabla ---
fig, ax = plt.subplots(figsize=(10,5))
ax.axis("off")
tbl = ax.table(cellText=horario_df.values,
               rowLabels=horario_df.index,
               colLabels=horario_df.columns,
               cellLoc="center", loc="center")
tbl.scale(1.2, 1.5)
plt.title("Horario Optimizado con Restricciones")
plt.show()
