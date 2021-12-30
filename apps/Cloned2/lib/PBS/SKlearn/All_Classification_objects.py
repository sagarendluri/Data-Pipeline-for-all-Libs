import joblib
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.svm import SVC
import eli5
from eli5.sklearn import PermutationImportance
from dateutil.parser import parse
from sklearn.neural_network import MLPClassifier


def RandomForest_Classifier(X_train, y_train, X_test, y_test, grids, target, model_):
    model_file_name = "SKlean_Class_Random_forest" + model_ + target + '.pkl'
    rfg = RandomizedSearchCV(RandomForestClassifier(), grids, cv=5, n_jobs=-1, verbose=True, refit=True)
    rfg.fit(X_train, y_train)
    train_accuracy = rfg.score(X_train, y_train)*100
    test_accuracy = rfg.score(X_test, y_test)*100
    y_pred = rfg.best_estimator_.predict(X_test)
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    y_test = y_test.reshape(len(y_test), 1)
    y_pred = y_pred.reshape(len(y_pred), 1)
    diff = (y_test - y_pred)
    mbe = diff.mean()
    print("train_accuracy,test_accuracy", train_accuracy, test_accuracy)
    grid = rfg.best_params_
    estimator = rfg.best_estimator_
    clsf_report = pd.DataFrame(
        classification_report(y_true=y_test, y_pred=y_pred, output_dict=True)).transpose()
    print(clsf_report)
    cm1 = confusion_matrix(y_test, y_pred)
    # Accuracy = metrics.accuracy_score(y_test, y_pred) * 100
    cm = {'confusion_metrics': cm1.tolist()}
    print(cm1)
    joblib.dump(estimator, model_file_name)
    importance = rfg.best_estimator_.feature_importances_
    for k, v in grid.items():
        try:
            if v == int:
                grid[k] = int(v)
                # grid['learning_rate'] == float(0.045)
        except ValueError:
            if v == float:
                grid[key] = float(value)
    importance = importance.astype(float)
    models = 'RandomForest_Classifier_'
    return train_accuracy, test_accuracy, y_pred, importance, grid, estimator, model_file_name, models, cm ,mbe


def XGBoost_Classifier(X_train, y_train, X_test, y_test, grids, target, model_):
    model_file_name = "SKlean_Class_XGBOOST" + model_ + target + '.pkl'
    xgb = RandomizedSearchCV(XGBClassifier(), grids, cv=5, n_jobs=-1, verbose=True, refit=True)
    xgb.fit(X_train, y_train)
    train_accuracy = xgb.score(X_train, y_train)*100
    test_accuracy = xgb.score(X_test, y_test)*100
    grid = xgb.best_params_
    estimator = xgb.best_estimator_
    y_pred = xgb.best_estimator_.predict(X_test)
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    y_test = y_test.reshape(len(y_test), 1)
    y_pred = y_pred.reshape(len(y_pred), 1)
    diff = (y_test - y_pred)
    mbe = diff.mean()
    clsf_report = pd.DataFrame(
        classification_report(y_true=y_test, y_pred=y_pred, output_dict=True)).transpose()
    print(clsf_report)
    cm1 = confusion_matrix(y_test, y_pred)
    # Accuracy = metrics.accuracy_score(y_test, y_pred) * 100
    cm = {'confusion_metrics': cm1.tolist()}
    joblib.dump(estimator, model_file_name)
    importance = xgb.best_estimator_.feature_importances_
    for k, v in grid.items():
        try:
            if v == int:
                grid[k] = int(v)
                # grid['learning_rate'] == float(0.045)
        except ValueError:
            if v == float:
                grid[key] = float(value)
    importance = importance.astype(float)
    models = 'XGBoost_Classifier'
    return train_accuracy, test_accuracy, y_pred, importance, grid, estimator, model_file_name, models, cm , mbe


def KNeighbors_Classifier(X_train, y_train, X_test, y_test, grids, target, model_):
    model_file_name = "SKlean_Class_KNeighbors_" + model_ +"_"+ target + '.pkl'
    knn = RandomizedSearchCV(KNeighborsClassifier(), grids, cv=5, n_jobs=-1, verbose=True, refit=True)
    knn.fit(X_train, y_train)
    train_accuracy = knn.score(X_train, y_train)*100
    test_accuracy = knn.score(X_test, y_test)*100
    print("train_accuracy,test_accuracy", train_accuracy, test_accuracy)
    y_pred = knn.best_estimator_.predict(X_test)
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    y_test = y_test.reshape(len(y_test), 1)
    y_pred = y_pred.reshape(len(y_pred), 1)
    diff = (y_test - y_pred)
    mbe = diff.mean()
    clsf_report = pd.DataFrame(
        classification_report(y_true=y_test, y_pred=y_pred, output_dict=True)).transpose()
    print(clsf_report)
    grid = knn.best_params_
    estimator = knn.best_estimator_
    cm1 = confusion_matrix(y_test, y_pred)
    cm = {'confusion_metrics': cm1.tolist()}
    joblib.dump(estimator, model_file_name)
    perm = PermutationImportance(knn, random_state=1).fit(X_train, y_train)
    importance = perm.feature_importances_
    for k, v in grid.items():
        grid[k] = int(v)
    models = 'KNeighbors_Classifier_'
    return train_accuracy, test_accuracy, y_pred, importance, grid, estimator, model_file_name, models, cm,mbe


def Multilayer_Perceptron_Classifier(X_train, y_train, X_test, y_test, grids, target, model_):
    model_file_name = "SKlean_Classifier_Multilayer_Perceptron_" + model_ + target + '.pkl'
    mlp = RandomizedSearchCV(MLPClassifier(), grids, cv=5, n_jobs=-1, verbose=True, refit=True)
    mlp.fit(X_train, y_train)
    train_accuracy = mlp.score(X_train, y_train)*100
    test_accuracy = mlp.score(X_test, y_test)*100
    y_pred = mlp.best_estimator_.predict(X_test)
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    y_test = y_test.reshape(len(y_test), 1)
    y_pred = y_pred.reshape(len(y_pred), 1)
    diff = (y_test - y_pred)
    mbe = diff.mean()
    clsf_report = pd.DataFrame(
        classification_report(y_true=y_test, y_pred=y_pred, output_dict=True)).transpose()
    print(clsf_report)
    grid = mlp.best_params_
    estimator = mlp.best_estimator_
    cm1 = confusion_matrix(y_test, y_pred)
    # Accuracy = metrics.accuracy_score(y_test, y_pred) * 100
    cm = {'confusion_metrics': cm1.tolist()}
    joblib.dump(estimator, model_file_name)
    perm = PermutationImportance(gd, random_state=1).fit(X_train, y_train)
    importance = perm.feature_importances_
    models = 'Multilayer_Perceptron_Classifier_'
    return train_accuracy, test_accuracy, y_pred, importance, grid, estimator, model_file_name, models, cm ,mbe


def SVC_Classifier(X_train, y_train, X_test, y_test, grids, target, model_):
    model_file_name = "SKlean_Classifier_Support_Vector_M_" + model_ + target + '.pkl'
    svm = RandomizedSearchCV(SVC(), grids, cv=5, n_jobs=-1, verbose=True, refit=True)
    svm.fit(X_train, y_train)
    train_accuracy = svm.score(X_train, y_train)*100
    test_accuracy = svm.score(X_test, y_test)*100
    y_pred = svm.best_estimator_.predict(X_test)
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    y_test = y_test.reshape(len(y_test), 1)
    y_pred = y_pred.reshape(len(y_pred), 1)
    diff = (y_test - y_pred)
    mbe = diff.mean()
    clsf_report = pd.DataFrame(
        classification_report(y_true=y_test, y_pred=y_pred, output_dict=True)).transpose()
    print(clsf_report)
    grid = svm.best_params_
    estimator = svm.best_estimator_
    cm1 = confusion_matrix(y_test, y_pred)
    # Accuracy = metrics.accuracy_score(y_test, y_pred) * 100
    cm = {'confusion_metrics': cm1.tolist()}
    joblib.dump(estimator, model_file_name)
    importance = svm.best_estimator_.coef_
    imp = importance.tolist()
    importance = imp[0]
    models = 'Support_Vector_Classifier_'
    return train_accuracy, test_accuracy, y_pred, importance, grid, estimator, model_file_name, models, cm ,mbe