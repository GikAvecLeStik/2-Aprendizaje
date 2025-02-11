import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from bosque_aleatorio import entrena_bosque, predice_bosque

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, delimiter=";")

X = data.drop(columns=["quality"]).values  # Features
y = data["quality"].values  # Target (classification problem)

y = np.where(y >= 6, 1, 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_trees_values = [5, 10, 20, 50]
max_depth_values = [None, 5, 10]
max_features_values = [2, int(np.sqrt(X.shape[1])), X.shape[1]]

for n_trees in num_trees_values:
    for max_depth in max_depth_values:
        for max_features in max_features_values:
            print(f"Entrenando bosques aleatorios con {n_trees} arboles, max_prof={max_depth}, max_atributos={max_features}")

            trees = entrena_bosque(X_train, y_train, M=n_trees, max_features=max_features)

            y_pred = predice_bosque(trees, X_test)

            acc = accuracy_score(y_test, y_pred)
            print(f"Exactitud: {acc:.4f}\n")
