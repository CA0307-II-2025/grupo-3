# 📆  Planificación

## 🎯  Objetivo del Sprint:

Lectura de base de datos y asegurar su consistencia para análisis.
Incluye completar la limpieza final de datos, documentar la estructura de la base, consolidar el entorno de trabajo compartido y avanzar con el marco teórico y bibliografía del reporte científico.


## 😃  Historias de usuario

- **HU#5 – “Script final de lectura y limpieza”** (Estimación: 5 pts) – *El script procesa, limpia/imputa valores faltantes, filtra outliers y genera la base final almacenada localmente.*
- **HU#6 – “Documentación de la base de datos”** (Estimación: 3 pts) – *Documento con estructura de datos, unidad estadística, localización espacial, temporal, etc.*
- **HU#7 – “Consolidar entorno de trabajo reproducible”** (Estimación: 3 pts) – *Todos los miembros del equipo pueden trabajar bajo el mismo entorno sin errores.*
- **HU#8 – “Escritura del marco teórico y enlace de la bibliografía”** (Estimación: 5 pts) – *Marco teórico redactado y referencias correctamente enlazadas en el reporte científico.*

---

## 🔜  Plan de alto nivel:
- *Semana 1:*
  - Implementar script final de limpieza y validación de la base de datos.
  - Documentar estructura de datos y preparar archivo con descripciones.

- *Semana 2:*
  - Validar consistencia del entorno de trabajo para todos los integrantes.
  - Redactar el marco teórico con bibliografía enlazada en el reporte científico.

---
## 🥇  Criterios de aceptación del Sprint:
- [ ] Todas las historias listadas completadas y aceptadas por el profesor.
- [x] Los datos procesados se almacenan localmente en formato estándar.
- [x] Scripts de limpieza eliminan o imputan valores faltantes y filtran outliers.
- [x] Documento con estructura de los datos correctamente entregado.
- [x] Reporte científico con marco teórico ordenado y referencias enlazadas.

---
## 📌  Asignación de tareas inicial
- *Jeikel:* Script final de lectura y limpieza + colaborar con B en documentación (ej. describir variables transformadas en el script).
- *Gabriel:* Marco teórico y bibliografía + redactar parte del reporte científico con la estructura ya establecida.
- *Andy:* Documentación de la base de datos + revisión de bibliografía (apoyar a D con fichas de papers)
- *Diego:* Configuración del entorno de trabajo compartido + pruebas de ejecución de los scripts en diferentes máquinas.
## 🚫 Posibles bloqueos o impedimentos conocidos

- **Bloqueo:** _No tenemos la base de datos limpia.
- **Solución** _Crear una historia solo para limpiar la base de datos_.




# ⏳  Daily

##  Fecha: 2025-08-27

### Jeikel:
- **¿Qué hice ayer?**: Realicé la lectura y limpieza de la base de datos, verificando que no existieran datos faltantes. Además, generé una tabla resumen de las variables numéricas con mínimos, cuantiles, máximos y promedios.
- **¿Qué haré hoy?**: Ajustar y dar formato a la tabla resumen para que sea clara y fácil de interpretar, e integrar estos resultados en la documentación del proyecto.
- **¿Hay algo que me está bloqueando?**: No tengo bloqueos en este momento; la base está limpia y la tabla de resumen ya se genera correctamente.

### Andy:
- **¿Qué hice ayer?**: Nada.
- **¿Qué haré hoy?**: Investigar
- **¿Hay algo que me está bloqueando?**: No.


##  Fecha: 2025-08-28

### Andy:
- **¿Qué hice ayer?**: Investigar.
- **¿Qué haré hoy?**: Añadir documentación de la base
- **¿Hay algo que me está bloqueando?**: No.

### Diego:
- **¿Qué hice ayer?**: Configuré mi entorno para poder ejecutar correctamente cualquier script .py sin depender del repositorio en VS Code
- **¿Qué haré hoy?**: Revisaré que todos los módulos, chequeando que sean ejecutables y no den ningun tipo de error.
- **¿Hay algo que me está bloqueando?**: La disponibilidad de tiempo ultimadamente es limitada




# 🔍   Revisión en clase (Fecha: YYYY-MM-DD)


## 📈  Resultado mostrado

- *Funcionalidad A:* (ej: "Carga automática de dataset desde CSV en base de datos completada").
- *Funcionalidad B:* (ej: "Gráficos descriptivos generados dinámicamente").


## :arrows_counterclockwise:  Retroalimentación

- **Profesor**:
- **Compañeros:**


## ✔️  Criterios de aceptación cumplidos:
- [] _Historias 1, 2, 3. completadas. Falta la historia 4.
- [x] Carga automática de la base de datos.


# 🔙  Retrospective – Fecha: YYYY-MM-DD

## :white_check_mark: Qué salió bien
1.  _Colaboración en el equipo_ Logramos terminar el sprint a tiempo.
1.  _Usamos commits convencionales correctamente y no hubo errores_
1.  Documentación actualizada al día evitó retrabajo luego.



## :no_good: Qué podría mejorar

- _Gestión de tiempo en Daily:_ a veces se extendieron a 20 min discutiendo detalles innecesarios.
- _Claridad de criterios de aceptación:_ En HU2 inicialmente no estaba claro cómo validar "datos limpios". Mejoraremos definición de *Done* para tareas de datos.
- _Distribución de carga:_ Persona A quedó sobrecargada con 3 historias. El próximo sprint se equilibrará asignación más temprano.


## :pencil: Acciones concretas  para el próximo sprint
1. **Timebox en Daily** – SM usará temporizador de 15 min y cortará discusiones largas, anotándolas para after.
2. **Refinar historias en refinamiento semanal** – Agregar criterios de aceptación más detallados, especialmente para historias técnicas (como limpieza de datos).
3. **Balancear asignación tareas** – Implementar mini-plan al inicio del sprint donde cada dev toma carga similar; SM monitoreará que nadie tenga >40% de tareas.
