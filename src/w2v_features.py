import re
from collections import Counter
from pathlib import Path
import json
from config import SPLITS_PATH


import numpy as np
import gensim.downloader as api
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

MODEL_NAME = "glove-wiki-gigaword-300"
DATA_DIR = Path(SPLITS_PATH).parent          # la carpeta data/

TOKEN_RE = re.compile(r"[a-z]+")    # Expresión regular que se usa para detectar palabras y eliminar contenido como espacios, comas, guiones o dígitos

def tokenize(text):
    """
    Se pasa el texto a minúsculas y se devuelven las palabras que encajan en la expresión regular.
    Se descartan las palabras de 1 o 2 letras y se eliminan las Stop Words (Ej: the, of, which...)
    """
    tokens = TOKEN_RE.findall(str(text).lower())
    return [t for t in tokens if len(t) > 2 and t not in ENGLISH_STOP_WORDS]

def doc_vector(tokens, kv):
    """
    Se seleccionan los vectores del modelo preentrenado para cada palabra (token) que se le pase.
    Se omiten las OOV.
    Si no hay vectores se devuelve una matriz de ceros.
    Si hay vectores hace la media por dimensiones (axis=0) y devuelve también cuántos tokens contribuyeron.
    La media cuenta cada aparición: si energy sale 5 veces, pesa 5 veces. Así, 
    las palabras repetidas dominan el vector del documento.
    """
    """Media de los vectores de las palabras conocidas.
    Devuelve (vector, nº de tokens usados)."""
    vecs = [kv[t] for t in tokens if t in kv]
    if len(vecs) == 0:
        return np.zeros(kv.vector_size, dtype=np.float32), 0
    return np.mean(vecs, axis=0), len(vecs)

def build_matrix(texts, kv):
    X = np.zeros((len(texts), kv.vector_size), dtype=np.float32)
    n_total = np.zeros(len(texts), dtype=int)   # tokens por documento
    n_used = np.zeros(len(texts), dtype=int)    # tokens con vector
    oov = Counter()
    for i, text in enumerate(texts):
        tokens = tokenize(text)
        X[i], n_used[i] = doc_vector(tokens, kv)
        n_total[i] = len(tokens)
        for t in tokens:
            if t not in kv:
                oov[t] += 1
    return X, n_total, n_used, oov

# ---------------------------------------------------------------------------
# Guardado y carga
# ---------------------------------------------------------------------------
def feature_paths(model_name=MODEL_NAME, data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    base = f"emb_{model_name}_mean"
    return {
        "X": data_dir / f"{base}.npy",
        "ids": data_dir / f"{base}_ids.npy",
        "stats": data_dir / f"{base}_stats.npz",
        "oov": data_dir / f"{base}_oov.json",
    }


def save_features(paths, X, ids, n_total, n_used, oov):
    paths["X"].parent.mkdir(parents=True, exist_ok=True)
    np.save(paths["X"], X)
    np.save(paths["ids"], np.asarray(ids))
    np.savez(paths["stats"], n_total=n_total, n_used=n_used)
    with open(paths["oov"], "w", encoding="utf-8") as f:
        json.dump(dict(oov), f, ensure_ascii=False)

def load_features(paths):
    X = np.load(paths["X"])
    ids = np.load(paths["ids"])
    stats = np.load(paths["stats"])
    with open(paths["oov"], encoding="utf-8") as f:
        oov = Counter(json.load(f))
    return X, ids, stats["n_total"], stats["n_used"], oov

def load_or_build(df, model_name=MODEL_NAME, data_dir=DATA_DIR, kv=None, force=False):
    """Devuelve (X, n_total, n_used, oov) para las filas de df, en su mismo orden.

    Si los ficheros existen y sus projectID coinciden con los de df, los carga
    (segundos). Si no, descarga el modelo, calcula la matriz y la guarda.
    kv: vectores ya cargados (opcional); si es None se descargan con gensim.
    force=True obliga a recalcular aunque existan los ficheros.
    """
    paths = feature_paths(model_name, data_dir)
    ids = df["projectID"].to_numpy()

    todos_existen = True
    for ruta in paths.values():
        if not ruta.exists():
            todos_existen = False

    if todos_existen and not force:
        X, saved_ids, n_total, n_used, oov = load_features(paths)
        if len(saved_ids) == len(ids) and (saved_ids == ids).all():
            print(f"Matriz cargada desde {paths['X']}")
            return X, n_total, n_used, oov
        print("Los projectID guardados no coinciden con df: se recalcula.")

    if kv is None:
        import gensim.downloader as api
        kv = api.load(model_name)   # se descarga solo la primera vez

    X, n_total, n_used, oov = build_matrix(df["summary"].fillna("").tolist(), kv)
    save_features(paths, X, ids, n_total, n_used, oov)
    print(f"Matriz calculada y guardada en {paths['X']}")
    return X, n_total, n_used, oov


def print_stats(n_total, n_used, oov):
    print("Documentos sin ningún token válido:", int((n_used == 0).sum()))
    print("Tokens con vector (global):", n_used.sum() / n_total.sum())
    print("Cobertura media por documento:", (n_used / np.maximum(n_total, 1)).mean())
    print("Tipos OOV distintos:", len(oov))
    print("OOV más frecuentes:", oov.most_common(20))

if __name__ == "__main__":
    from labels import normalizar_datos 
    df, _ = normalizar_datos()
    # splits = np.load(SPLITS_PATH)
    # df es el DataFrame filtrado de labels.py
    # X, n_total, n_used, oov = build_matrix(df["summary"].fillna("").tolist(), kv)
    X, n_total, n_used, oov = load_or_build(df)
    print("Forma de la matriz:", X.shape)
    print_stats(n_total, n_used, oov)
    """
    Salida:
    Documentos sin ningún token válido: 1
    Tokens con vector (global): 0.974237223402856
    Cobertura media por documento: 0.9740686581236958
    Tipos OOV distintos: 51157
    OOV más frecuentes: [('esrs', 1070), ('crispr', 472), ('intersectoral', 427), ('secondments', 425), ('covid', 310),
    ('msca', 258), ('blockchain', 258), ('bioeconomy', 250), ('organoids', 247), ('spatio', 202), ('valorisation', 195),
    ('transcriptomics', 189), ('exascale', 177), ('transcriptomic', 164), ('perovskites', 155), ('phenotyping', 150),
    ('eimc', 145), ('optogenetic', 135), ('neuromorphic', 133), ('optogenetics', 131)]
    """