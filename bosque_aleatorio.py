import numpy as np
from arboles_numericos import entrena_arbol
from collections import Counter

def entrena_bosque(X, y, M=10, sample_size=None, max_features=None, random_state=None):
    """

    :param X (numpy.ndarray): matriz de datos
    :param y (numpy.ndarray): valor del atributo de entrenado
    :param n_trees (int): numero de trees usados
    :param sample_size (int): numero de ejemplos de entrenamiento por cada arbol
    :param max_features: numero de atributos por cada arbol
    :param random_state: estado aleatorio
    :return:
        list : una lista de arboles entrenados
    """
    if random_state is None:
        np.random.seed(random_state)


    n_samples, n_features = X.shape
    sample_size = sample_size if sample_size else n_samples
    max_features = max_features if max_features else int(np.sqrt(n_features))

    trees = []

    for _ in range(M):
        indices = np.random.choice(n_samples, sample_size, replace=True)
        X_sample, y_sample = X[indices], y[indices]

        feature_indices = np.random.choice(n_features, size=max_features, replace=False)
        X_sample = X_sample[:,feature_indices]

        tree = entrena_arbol(X_sample, y_sample)
        trees.append((tree, feature_indices))

    return trees

def predice_bosque(trees, X):
    """
    :param trees (list): lista de arboles entrenados
    :param X (numpy.ndarray): matriz de atributos para las predicciones
    :return (numpy.ndarray): predicciones para cada muestra
    """
    all_predicciones = np.array([tree.predict(X) for tree in trees])

    if np.issubdtype(all_predicciones.dtype, np.floating):
        predicciones = [Counter(all_predicciones[:,i]).most_common(1)[0][0] for i in range(X.shape[0])]

    else:
        predicciones = np.mean(all_predicciones, axis=0)

    return np.array(predicciones)




