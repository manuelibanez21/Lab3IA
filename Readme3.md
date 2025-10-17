# 🧠 Proyecto de Optimización de Horarios con Algoritmos Genéticos

Este proyecto implementa un **algoritmo genético** para optimizar horarios de clases considerando:
- Disponibilidad de profesores  
- Preferencias de materias por horario  
- Conflictos de grupos y profesores  
- Maximización del "fitness" de los horarios generados.

---

## 📌 Descripción General

Este programa genera horarios académicos de forma automática aplicando **técnicas evolutivas** para mejorar progresivamente la asignación de clases, profesores y materias.

Se define:
- Un conjunto de **profesores** con horarios disponibles.  
- Un conjunto de **materias** con horarios preferidos.  
- Varios **grupos de estudiantes**.  
- **Intervalos de tiempo** para programar las clases.

A partir de esto, el algoritmo busca un horario que:
- Respete la disponibilidad de profesores.  
- Cumpla con las preferencias horarias de materias.  
- Evite choques entre grupos y profesores.  
- Maximice una función de aptitud (**fitness**).

---

## 🧾 Estructura del Código

### 1. **Datos iniciales y restricciones**

```python
self.teachers = { ... }        # Profesores y sus horarios disponibles
self.subjects = [ ... ]        # Lista de materias
self.groups = [ ... ]          # Lista de grupos
self.time_slots = [ ... ]      # Lista de horarios disponibles
self.preferences = { ... }     # Preferencias horarias por materia
Estos diccionarios y listas definen las reglas iniciales del problema.

2. Creación de un horario inicial
python
def create_schedule(self):
    """Crea un horario aleatorio válido con disponibilidad de profes"""
    ...
Asigna aleatoriamente materias y profesores a cada grupo en cada horario.

Esta es la población inicial para el algoritmo genético.

3. Evaluación del horario (Fitness)
python
def evaluate_schedule(self, schedule):
    """Calcula el fitness basado en restricciones"""
    ...
El fitness premia:

Que no haya choques de horarios entre grupos o profesores.

Que se respeten disponibilidades y preferencias.

Penaliza:

Choques de horarios.

Asignaciones fuera de disponibilidad.

Materias en horarios no preferidos.

4. Operadores genéticos
Mutación: Cambia aleatoriamente el profesor o la materia de una clase.

python
def mutation(self, schedule):
    ...
Cruce: Combina dos horarios (padres) para generar un nuevo (hijo).

python
def crossover(self, s1, s2):
    ...
5. Algoritmo Genético Principal
python
def genetic_algorithm(self, pop_size=50, generations=100, mutation_rate=0.2):
    ...
Pasos:

Crear población inicial de horarios.

Evaluar fitness de cada horario.

Seleccionar mejores horarios.

Aplicar cruce y mutación para generar nuevos.

Repetir por varias generaciones.

Guardar el mejor horario encontrado.

También se grafica la convergencia del fitness a lo largo de las generaciones.

📊 Resultados
Después de ejecutar el algoritmo:

python
optimizer = ScheduleOptimizer()
best_horario, best_score = optimizer.genetic_algorithm()
Se obtiene:

✅ El mejor horario encontrado.

📈 La evolución del fitness a través de las generaciones.

🧾 Una tabla legible del horario final para todos los grupos.

Ejemplo de salida en consola:

text
Mejor horario encontrado (fitness: 85)
('Lun-9:00', 'Grupo1', 'ProfA', 'Matemáticas')
('Lun-9:00', 'Grupo2', 'ProfB', 'Arte')
...
🧮 Representación en Tabla
python
horario_df = pd.DataFrame(index=optimizer.groups, columns=optimizer.time_slots)
Se muestra un horario final organizado por grupos y horarios, con la asignación de materia y profesor.

Lun-9:00	Lun-11:00	Mar-9:00	...
Grupo1	Matemáticas-ProfA	Historia-ProfB	Ciencias-ProfC	...
Grupo2	Arte-ProfB	Arte-ProfC	...	...
Grupo3	...	...	...	...
También se genera una tabla gráfica usando matplotlib:
