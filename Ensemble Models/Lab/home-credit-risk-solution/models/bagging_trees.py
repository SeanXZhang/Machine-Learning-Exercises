import os.path

from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier

from config import *
from utils.utils import *


def bagging_decision_trees(data_train, y_train, data_test, y_test):
    clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(class_weight='balanced'))

    print('saving model to file...')
    save_binary(clf, os.path.join(RESULTS_MODEL_SUB, f'bagging_decision_trees.clf'))

    prediction = clf.fit(data_train, y_train).predict(data_test)
    score = evaluate_prediction(prediction, y_test)
    print(score)
    return prediction


def param_tuning(data_train, y_train, data_test, y_test):
    clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(class_weight='balanced'))

    search_params = {
        'n_estimators': np.arange(10, 100, 10),
        'max_samples': [0.05, 0.1, 0.2, 0.5]
    }

    n_iter_search = 20
    cv = RandomizedSearchCV(clf, param_distributions=search_params, n_iter=n_iter_search,
                            scoring={'score': 'f1'}, refit='score')  # accuracy is not a good metric for imbalanced dataset

    start = time.time()
    cv.fit(data_train, y_train)

    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time.time() - start), n_iter_search))
    print(cv.best_params_)
    #report(cv.cv_results_) #check the detailed model selection report

    best_clf = cv.best_estimator_

    # by the end of randomized search, the model will be refitted on the entire data set
    print('saving model to file...')
    save_binary(best_clf, os.path.join(RESULTS_MODEL_SUB, f'bagging_decision_trees_best_params.clf'))

    prediction = best_clf.predict(data_test)
    score = evaluate_prediction(prediction, y_test)
    print(score)
    return prediction



