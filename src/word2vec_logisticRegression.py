import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from eval import evaluate, save_results
from config import SPLITS_PATH
from labels import normalizar_datos
from w2v_features import load_or_build

df, _ = normalizar_datos()
X, n_total, n_used, oov = load_or_build(df)

s = np.load(SPLITS_PATH)
assert (df["projectID"].to_numpy() == s["project_ids"]).all()   # mismo df que A
train_idx, val_idx, test_idx = s["train_idx"], s["val_idx"], s["test_idx"]

y = df["y2"].to_numpy()
scaler = StandardScaler().fit(X[train_idx])        # SOLO con train
X_train = scaler.transform(X[train_idx])
X_val = scaler.transform(X[val_idx])
X_test = scaler.transform(X[test_idx])
y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]



clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X_train, y_train)

pred_val = clf.predict(X_val)
print(evaluate(y_val, pred_val)["macro_f1"])