"""
eval.py - Evaluación común y guardado de resultados.

Todos los modelos (TF-IDF, Word2Vec, BERT...) guardan sus resultados con
estas dos funciones, para que las comparaciones sean homogéneas:

    metrics = evaluate(y_true, y_pred)
    save_results("tfidf", "logreg", 2, "val", ids, y_true, y_pred)

Salidas (carpeta RESULTS_DIR de config.py):
    metrics.csv                              una fila por (representacion, modelo, nivel, split)
    preds_<representacion>_<modelo>_<nivel>.csv   columnas: id, y_true, y_pred, split [, proba_<clase>]

Se hace una evaluación para el nivel 1 y otra para el nivel 2 de la jerarquía de etiquetas
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from config import RESULTS_DIR

METRICS_COLUMNS = [
    "representacion", "modelo", "nivel", "split",
    "macro_f1", "accuracy", "ic_low", "ic_high",
]
KEY_COLUMNS = ["representacion", "modelo", "nivel", "split"]
VALID_SPLITS = ("val", "test")


def evaluate(y_true, y_pred, labels=None):
    """Calcula macro-F1, accuracy y F1 por clase.

    labels: lista de clases a considerar. Si es None, se usa la unión de las
    clases presentes en y_true e y_pred. Las clases sin ningún acierto cuentan
    como F1 = 0 (zero_division=0), como es habitual en macro-F1.

    Devuelve un diccionario:
        {"macro_f1": float, "accuracy": float, "f1_per_class": {clase: f1}}
    """
    y_true = np.asarray(y_true)     # Se convierte a lista
    y_pred = np.asarray(y_pred)     # Se convierte a lista
    if len(y_true) != len(y_pred):
        raise ValueError("y_true e y_pred deben tener la misma longitud")

    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    """
    Se calculan "accuracy", F1 macro y F1 para cada clase tal y como pide el enunciado.
    Si solo se caulculara la "accuracy" con clases muy desbalanceadas el resultado podría llevar
    a confusión y estar valorando un modelo como bueno simplemente porque se predice la clase mayoritaria.
    De esta forma con F1 se da la misma importancia a todas las clases.
    """

    """
    Cuando una clase nunca se predice, su precisión sería 0/0. sklearn por defecto avisa con un warning y 
    devuelve 0. Con zero_division=0 se devuelve 0 sin warning, y el F1 de esa clase queda en 0.
    """
    macro_f1 = f1_score(y_true, y_pred, labels=labels,
                        average="macro", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    f1_each = f1_score(y_true, y_pred, labels=labels,
                       average=None, zero_division=0)

    f1_per_class = {}
    for clase, valor in zip(labels, f1_each):
        f1_per_class[int(clase)] = float(valor)

    return {
        "macro_f1": float(macro_f1),
        "accuracy": float(accuracy),
        "f1_per_class": f1_per_class,
    }


def _replace_rows(existing, new_rows, key_columns):
    """Devuelve existing sin las filas con la misma clave que new_rows, más new_rows.

    Así, volver a ejecutar un experimento sustituye sus resultados en lugar de
    duplicarlos.
    """
    if existing is None or existing.empty:
        return new_rows.reset_index(drop=True)

    keep = pd.Series(True, index=existing.index)
    for _, fila in new_rows[key_columns].drop_duplicates().iterrows():
        coincide = pd.Series(True, index=existing.index)
        for col in key_columns:
            coincide &= existing[col] == fila[col]
        keep &= ~coincide

    return pd.concat([existing[keep], new_rows], ignore_index=True)


def _read_csv_if_exists(path):
    if Path(path).exists():
        return pd.read_csv(path)
    return None


def save_results(representacion, modelo, nivel, split, ids, y_true, y_pred,results_dir=None):
    """Guarda las predicciones y añade/actualiza la fila de metrics.csv.

    representacion, modelo: cadenas sin espacios (p. ej. "tfidf", "logreg").
    nivel: 1 o 2.
    split: "val" o "test".
    ids: identificadores de los proyectos (projectID), mismo orden que y_true.
    Devuelve el diccionario de evaluate().
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"split debe ser uno de {VALID_SPLITS}, no '{split}'")

    ids = np.asarray(ids)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if not (len(ids) == len(y_true) == len(y_pred)):
        raise ValueError("ids, y_true e y_pred deben tener la misma longitud")

    out_dir = Path(results_dir) if results_dir is not None else Path(RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- métricas ---
    metrics = evaluate(y_true, y_pred)

    # --- predicciones ---
    """
    Esta parte del código se encarga de guardar las predicciones que hasta el momento están en una variable en la memoria, en un fichero
    de salida "out_dir / f"preds_{representacion}_{modelo}_{nivel}.csv". Primero se comprueba si el fichero existe o no. Si el fichero
    no existe se ejecuta la rama else y se guarda. Si el fichero existe se actualizan las filas correspondientes al split que se esté
    ejecutando. Por ejemplo:
    Si ejecuto el split "test", las filas de "test" se actualizan pero las de "val" se quedan como están de una ejecucuón anterior.
    """
    preds = pd.DataFrame({
        "id": ids,
        "y_true": y_true,
        "y_pred": y_pred,
        "split": split,
    })

    preds_path = out_dir / f"preds_{representacion}_{modelo}_{nivel}.csv"
    existing_preds = _read_csv_if_exists(preds_path)
    if existing_preds is not None:
        # se sustituyen solo las filas del mismo split (val o test)
        existing_preds = existing_preds[existing_preds["split"] != split]
        preds_all = pd.concat([existing_preds, preds], ignore_index=True)
    else:
        preds_all = preds
    preds_all.to_csv(preds_path, index=False)

    # --- fila de metrics.csv ---
    """
    En esta parte se guarda el csv con el resultado de la evaluación
    """
    row = pd.DataFrame([{
        "representacion": representacion,
        "modelo": modelo,
        "nivel": nivel,
        "split": split,
        "macro_f1": metrics["macro_f1"],
        "accuracy": metrics["accuracy"],
    }], columns=METRICS_COLUMNS)

    metrics_path = out_dir / "metrics.csv"
    existing_metrics = _read_csv_if_exists(metrics_path)
    metrics_all = _replace_rows(existing_metrics, row, KEY_COLUMNS)
    metrics_all.to_csv(metrics_path, index=False)

    return metrics


# ---------------------------------------------------------------------------
# Baseline de clase mayoritaria
# ---------------------------------------------------------------------------
def majority_class(y_train):
    """Clase más frecuente en el conjunto de entrenamiento."""
    valores, cuentas = np.unique(np.asarray(y_train), return_counts=True)   # cuenta cuantas veces aparece cada clase con return_counts=True
    return valores[np.argmax(cuentas)]


def run_baseline(df, splits, nivel=2, results_dir=None):
    """Evalúa el baseline en val y test y guarda los resultados.

    df: DataFrame filtrado de labels.normalizar_datos() (con projectID, y1, y2).
    splits: objeto cargado con np.load(SPLITS_PATH).
    """
    col = "y2" if nivel == 2 else "y1"
    y = df[col].to_numpy()
    ids = df["projectID"].to_numpy()

    mayoritaria = majority_class(y[splits["train_idx"]])   # solo con train
    print(f"Nivel {nivel}: clase mayoritaria (train) = {mayoritaria}")

    for split, clave in (("val", "val_idx"), ("test", "test_idx")):
        """ 
        split = val o test
        clave = val_idx o test_idx
        split es el nombre que se escribirá en metrics.csv y en el CSV de predicciones
        clave es el nombre del array dentro de splits.npz
        """
        idx = splits[clave]
        y_true = y[idx]
        y_pred = np.full(len(idx), mayoritaria) # crea un array con len(idx) en el que todos los elementos son iguales a la clase mayoritaria
        m = save_results("baseline", "mayoritaria", nivel, split,
                         ids[idx], y_true, y_pred, results_dir=results_dir)
        print(f"  {split}: macro-F1 = {m['macro_f1']:.4f}, accuracy = {m['accuracy']:.4f}")


if __name__ == "__main__":
    from config import SPLITS_PATH
    from labels import normalizar_datos

    df, _ = normalizar_datos()
    splits = np.load(SPLITS_PATH)
    # Se hace la prueba con el baseline
    for nivel in (2, 1):
        run_baseline(df, splits, nivel=nivel)