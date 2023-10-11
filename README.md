import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [6, 7]])  # Признаки
y = np.array([0, 0, 1, 1, 1])  # Метки классов

def minkowski_distance(x1, x2, p):
    return np.power(np.sum(np.abs(x1 - x2) ** p), 1/p)

def minkowski_predict(X_train, y_train, x_test, k, p):
    distances = [minkowski_distance(x_test, x, p) for x in X_train]
    k_indices = np.argsort(distances)[:k]
    k_nearest_labels = [y_train[i] for i in k_indices]
    most_common = np.bincount(k_nearest_labels).argmax()
    return most_common

new_point = np.array([4, 5])
k = 3  # Количество ближайших соседей
p = 2  # Параметр p для метода Минковского (2 для евклидова расстояния)
predicted_class = minkowski_predict(X, y, new_point, k, p)
print(f"Предсказанный класс для новой точки: {predicted_class}")
