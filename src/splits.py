# splits es para crearnos la partición train / validacion / test y guardarla en disco
# todos los modelos tenemos que entrenarlos y evaluarlos con los MISMOS PROYECTOS, para que sean validaciones buenas

import sys
sys.path.insert(0, "src") # para poder importar config y labels si ejecutamos desde raiz
import numpy as np
from sklearn.model_selection import train_test_split
from config import SEED, SPLITS_PATH # seed fija el azar, así tenemos el mismo result en cada ejec
from labels import normalizar_datos

# CARGAMOS LOS DATOS FILTRADOS "y" con labels y1, y2
# aqui no uso el valor de nº de proyectos, por eso voy a poner _ (guardará que tengo 35378 proyectos pero no lo uso)
df, _ = normalizar_datos()

# LOS INDICES VAN A SER LAS POSICIONES (0,1, ETC) DE CADA PROYECTO EN DF
# PARTO ESOS Nº, NO LOS DATOS
idx = np.arange(len(df))

# PARITICION 70 train, 30 validacion + test
train_idx, tmp_idx = train_test_split(
    idx, test_size= 0.3, stratify=df["y2"], random_state=SEED)
    #stratify mantine en cada parte la misma proporc. de cda clase, importante por el desbalanceo

# PARTICION, EL 30 lo divido en 15 para val y 15 para test
# así elijo los hiperparm, y test no lo toco hasta el final
valid_idx, test_idx = train_test_split(
    tmp_idx, test_size=0.5, stratify=df["y2"].iloc[tmp_idx], random_state=SEED)

# GUARDO LOS TRES CONJUNTOS en el .npz
np.savez(SPLITS_PATH, 
         train_idx=train_idx, val_idx=valid_idx, test_idx=test_idx,
         project_ids=df["projectID"].to_numpy())
        # project_ids es para comprobar, por si cargamos datos de otra forma comparamos con projectID 
        # con estos y vemos si los ind. coinciden

print("train:", len(train_idx), "valid:", len(valid_idx), "test:", len(test_idx))