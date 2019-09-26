from sklearn.neighbors import KNeighborsClassifier as KNN


def model_fit(k, x, y):
    knn = KNN(n_neighbors=k)
    knn.fit(x, y)
    return knn


def model_pred(model, x):
    predictions = model.predict(x)
    return predictions


def model_score(model, x, y):
    performance = model.score(x, y)
    return performance
