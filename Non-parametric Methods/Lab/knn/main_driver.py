from preprocess import *
from model import *
from general import *


def full_train():
    log_info('Load data from {}'.format(train_abs_path))
    dataset = load_dataset(train_abs_path)
    x_train, x_test, y_train, y_test = split_data(dataset)
    x_train = scaling(x_train, train=True)

    knn_model = model_fit(k, x_train, y_train)
    log_info('Model saved to results folder')
    pickle_object(knn_model, 'knn_{}_model.pkl'.format(k))

    train_results = model_pred(knn_model, x_train)
    pickle_object(train_results, 'knn_train_predictions.pkl')
    test_results = model_pred(knn_model, x_test)
    pickle_object(test_results, 'knn_test_predictions.pkl')

    performance_metrics = {'knn_{}'.format(k): model_score(knn_model, x_test, y_test)}
    log_info('Performance saved to results folder')
    pickle_object(performance_metrics, "metrics.pkl")


if __name__ == "__main__":
    folder_abs_path = os.getcwd()
    train_abs_path = os.path.join(folder_abs_path, 'data/iris.csv')
    k = 5

    full_train()



