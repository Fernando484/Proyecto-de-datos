# quiero ver por qué tenemos 32 proyectos de más y no nos está dando el 90.31%

# =============================================================================
# VERIFICACIÓN DE LA ETIQUETA (labels.py)
#
# PROBLEMA
#   El enunciado dice que, tras construir la etiqueta de nivel 2, se conserva
#   aproximadamente el 90,31 % de los proyectos. Con labels.py obtenemos
#   90,401 %: 31.982 de 35.378 proyectos, unos 32 más de los esperados (~31.950).
#
# QUÉ HEMOS COMPROBADO Y QUÉ HEMOS DESCARTADO
#   1. Estructura de SciVocCodes.xlsx: full_code es la ruta raíz -> código
#      (incluido él mismo). p[0] = nivel 1, p[1] = nivel 2, así que la regla
#      len(p) >= 2 es correcta.
#   2. Códigos repetidos en SciVocCodes: no hay (1010 filas, 1010 únicos),
#      así que el diccionario `path` no pisa rutas.
#   3. Desglose de los 35.378 proyectos:
#        - sin códigos (euroSciVocCode vacío) .... 3.326
#        - solo llegan a nivel 1 (se descartan) ..    70
#        - válidos (llegan a nivel 2) ............ 31.982
#      No hay projectID repetidos y ningún código de proyecto falta en SciVoc.
#   4. Regla estricta (exigir un código de nivel 3 o más): deja 31.769
#      (89,8 %), peor que la actual. Se descarta: un código que ya es de nivel 2
#      también es válido.
#   5. Número de clases: nivel 1 = 6 (esperado 6); nivel 2 = 39 (esperado
#      hasta 41). La etiqueta es coherente con el enunciado.
#
# DECISIÓN
#   No hemos encontrado la causa de los ~32 proyectos de diferencia, y todo lo
#   demás cuadra. Damos la etiqueta por buena y seguimos.
#   - Se anota en el README la cifra exacta (90,401 %) como desviación pequeña.
#   - Se consulta al profesor si es aceptable.
#   - Si hubiera que corregirla, solo hay que regenerar data/splits.npz.
#
# AVISO PARA EL MODELADO
#   La clase más pequeña de nivel 2 tiene solo 9 proyectos. Con 70/15/15 le
#   tocan ~6 en train, 1 en validación y 1-2 en test, así que sus métricas en
#   test serán muy ruidosas. Hay que comentarlo en el README.
# =============================================================================

import pandas as pd
sv = pd.read_excel("data/SciVocCodes.xlsx")
#prueba para ver si la jerarquía está bien montada 
#print(sv.columns.tolist())
#print(sv.head(10).to_string())

#prueba para ver si hay códigos repetidos
#sv = sv.dropna(subset=["code"])   # Elimina las filas sin código
#print("filas:", len(sv), "| códigos únicos:", sv["code"].nunique())
#dup = sv[sv["code"].duplicated(keep=False)].sort_values("code")
#print("filas con codigo repe:", len(dup)) 
#print(dup[["code", "full_code"]].head(20).to_string())

# voy a  mirar los grupos validos, que llegan al nivel 2 a ver cuántos hay, si hay repes o sin codigos...
import sys
sys.path.insert(0, "src")
from config import DATA_PATH, SCICODES_PATH
from labels import parse_codes

df = pd.read_excel(DATA_PATH)
sv = pd.read_excel(SCICODES_PATH).dropna(subset=["code"])
path = {int(c): [int(p) for p in fc.split("/") if p]
        for c, fc in zip(sv["code"], sv["full_code"])}

codes = df["euroSciVocCode"].apply(parse_codes)

print("total de proyectos:", len(df))
print("projectID repetidos:", df["projectID"].duplicated().sum())
print("sin codigos:", (codes.str.len() == 0).sum())

def estado(cs):
    ps = [path.get(c) for c in cs]
    if any(p is not None and len(p) >= 2 for p in ps):
        return "valido"
    if all(p is None for p in ps):
        return "ningun codigo en SciVoc"
    return "solo nivel 1"

print(codes[codes.str.len() > 0].apply(estado).value_counts())

# me ha salido:
#total de proyectos: 35378
#projectID repetidos: 0
#sin codigos: 3326
#euroSciVocCode
#valido          31982
#solo nivel 1       70
#Name: count, dtype: int64

# profundidad del primer codigo valido de cada proyecto
def prof_primer_valido(cs):
    for c in cs:
        p = path.get(c)
        if p is not None and len(p) >= 2:
            return len(p)
    return None

prof = codes.apply(prof_primer_valido).dropna().astype(int)
print("profundidad del primer codigo valido:")
print(prof.value_counts().sort_index())

# proyectos que tienen AL MENOS un codigo por debajo del nivel 2 (len >= 3)
estricto = codes.apply(lambda cs: any((path.get(c) or []) and len(path[c]) >= 3 for c in cs)).sum()
print("validos con regla estricta (algun codigo de nivel 3 o mas):", estricto)

# ahora veo :
# total de proyectos: 35378
#projectID repetidos: 0
#sin codigos: 3326
#euroSciVocCode
#valido          31982
#solo nivel 1       70
#Name: count, dtype: int64
#profundidad del primer codigo valido:
#euroSciVocCode
#2      661
#3     8631
#4    16180
#5     5366
#6      888
#7      256
#Name: count, dtype: int64
#validos con regla estricta (algun codigo de nivel 3 o mas): 31769

from labels import normalizar_datos
df_lab, n_antes = normalizar_datos()
print("clases nivel 1 (esperado 6):", df_lab["y1"].nunique(), sorted(df_lab["y1"].unique()))
print("clases nivel 2 (esperado hasta 41):", df_lab["y2"].nunique())
print(df_lab["y2"].value_counts().tail(8))
