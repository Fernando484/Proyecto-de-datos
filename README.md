# Proyecto-de-datos
Proyecto de la asignatura de Tratamiento de Datos

## 1. Datos y construcción de la etiqueta

### 1.1 Datos
El conjunto original contiene 35.378 proyectos H2020 con 19 variables. El
texto de entrada es `summary` (y `title`); la etiqueta se obtiene de
`euroSciVocCode`.

### 1.2 Construcción de la etiqueta
Siguiendo el enunciado, la etiqueta es el ancestro de **segundo nivel** de la
jerarquía EuroSciVoc del **primer** código de cada proyecto que llegue a ese
nivel (`src/labels.py`). Los proyectos se reparten así:

| Grupo | Proyectos |
|---|---|
| Sin ningún código (`euroSciVocCode` vacío) | 3.326 |
| Con códigos, pero ninguno llega a nivel 2 | 70 |
| **Con etiqueta de nivel 2** | **31.982** |
| Total | 35.378 |

Se conserva el **90,401 %** de los proyectos, frente al ≈ 90,31 % que indica
el enunciado (algunos proyectos más). Comprobamos la jerarquía de
`SciVocCodes.xlsx`, los códigos repetidos, los códigos ausentes y una regla
más estricta para el nivel 2, sin encontrar la causa. Las clases son las
esperadas (6 en el nivel 1 y 39 en el nivel 2), por lo que damos la etiqueta
por válida. Detalle en `notebooks/comprobacion_labels.ipynb`.

### 1.3 Partición train / validación / test
Partición estratificada por la clase de nivel 2, con semilla fija (42) y
proporciones 70/15/15, guardada en `data/splits.npz` para que todos los
modelos usen los mismos proyectos: **22387** de entrenamiento, **4797** de
validación y **4798** de test (lo hemos comprobado al ejecutar splits.py). La clase más pequeña tiene solo 9 proyectos,
así que sus métricas en test serán muy ruidosas.

