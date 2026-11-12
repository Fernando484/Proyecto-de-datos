from config import SEED, SCICODES_PATH, DATA_PATH
import pandas as pd
import ast

def load_data_pandas(path):
    df = pd.read_excel(path)
    return df

def parse_codes(x):
    if isinstance(x, (list, tuple)):
        return [int(c) for c in x]
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return []
        try:
            val = ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return []                      # texto mal formado
        if isinstance(val, (int, float)):  # un solo código sin corchetes
            val = [val]
        return [int(c) for c in val]
    return []                              # NaN u otro tipo

def pick_level2(codes, path):
    for c in codes:
        p = path.get(c)
        if p is not None and len(p) >= 2:
            return p[0], p[1]          # (nivel 1, nivel 2) del mismo código
    return None, None

def to_series(codes, path):
    return pd.Series(pick_level2(codes, path))


def normalizar_datos():
    """
    Cargar las bases de datos
    """
    projects_data = load_data_pandas(DATA_PATH)
    SciVocCodes_data = load_data_pandas(SCICODES_PATH)
    # La columna de códigos es un str no un array, se convierte en un array
    projects_data["euroSciVocCode"] = projects_data["euroSciVocCode"].apply(parse_codes)
    scivoc = SciVocCodes_data.dropna(subset=["code"])   # Elimina las filas sin código

    # Se crea el diccionario path con la clave "code" y el valor una lista con los "full_code" separados por /
    path = {int(c): [int(p) for p in fc.split("/") if p]
        for c, fc in zip(scivoc["code"], scivoc["full_code"])}
    # Se seleccionan los proyectos con nivel de jerarquía mayor o igual a 2
    pairs = [pick_level2(codes, path) for codes in projects_data["euroSciVocCode"]]
    projects_data["y1"] = [p[0] for p in pairs]     # Nivel 1 de jerarquía
    projects_data["y2"] = [p[1] for p in pairs]     # Nivel 2 de jerarquía

    n_before = len(projects_data)
    df = projects_data.dropna(subset=["y2"]).reset_index(drop=True)     # Se eliminan los proyectos que no llegan al nivel 2
    df["y1"] = df["y1"].astype(int)
    df["y2"] = df["y2"].astype(int)

    
    return df, n_before

if __name__ == "__main__":
    df, n_before = normalizar_datos()
    # print(f"Datos de los proyectos:\n {projects_data.head()}")
    # print(f"Datos de los códigos:\n {SciVocCodes_data.head()}")
    print(f"Los datos originales contienen {n_before} de los cuales se descartan {n_before - len(df)} proyectos por no contener etiquetas de nivel 2 o por falta de datos.\n Quedando el {(len(df) / n_before*100):.3f} % de los datos.")
    # print(n_before, len(df), len(df) / n_before)   # objetivo ≈ 0.9031