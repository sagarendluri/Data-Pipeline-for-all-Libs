import joblib
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,  classification_report ,confusion_matrix
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
def RandomForest_Classifier(X_train,y_train,X_test,y_test,grids ,target):
    model_file_name = "RandomForest_Classifier_" + target + '.pkl'
    gd = RandomizedSearchCV(RandomForestClassifier(), grids, cv=5, n_jobs=-1, verbose=True, refit=True)
    gd.fit(X_train, y_train)
    grid = gd.best_params_
    estimator = gd.best_estimator_
    y_pred = gd.predict(X_test)
    cm1 = confusion_matrix(y_test, y_pred)
    Accuracy = metrics.accuracy_score(y_test, y_pred) * 100
    cm = {'confusion_metrics': cm1.tolist()}
    joblib.dump(estimator, model_file_name)
    importance = gd.best_estimator_.feature_importances_
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
    return Accuracy,y_pred,importance,grid,estimator,model_file_name,models,cm
def XGBoost_Classifier(X_train,y_train,X_test,y_test,grids ,target):
    model_file_name = "XGBoost_Classifier_" + target + '.pkl'
    gd = RandomizedSearchCV(XGBClassifier(), grids, cv=5, n_jobs=-1, verbose=True, refit=True)
    gd.fit(X_train, y_train)
    grid = gd.best_params_
    estimator = gd.best_estimator_
    y_pred = gd.predict(X_test)
    cm1 = confusion_matrix(y_test, y_pred)
    Accuracy = metrics.accuracy_score(y_test, y_pred) * 100
    cm = {'confusion_metrics': cm1.tolist()}
    joblib.dump(estimator, model_file_name)
    importance = gd.best_estimator_.feature_importances_
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
    return Accuracy,y_pred,importance,grid,estimator,model_file_name,models,cm

def KNeighbors_Classifier(X_train,y_train,X_test,y_test,grids ,target):
    model_file_name = "KNeighbors_Classifier_" + target + '.pkl'
    gd = RandomizedSearchCV(KNeighborsClassifier(), grids, cv=5, n_jobs=-1, verbose=True, refit=True)
    gd.fit(X_train, y_train)
    grid = gd.best_params_
    estimator = gd.best_estimator_
    y_pred = gd.predict(X_test)
    cm1 = confusion_matrix(y_test, y_pred)
    Accuracy = metrics.accuracy_score(y_test, y_pred) * 100
    cm = {'confusion_metrics': cm1.tolist()}
    joblib.dump(estimator, model_file_name)
    perm = PermutationImportance(gd, random_state=1).fit(X_train, y_train)
    importance = perm.feature_importances_
    for k, v in grid.items():
        grid[k] = int(v)
    models = 'KNeighbors_Classifier_'
    return Accuracy,y_pred,importance,grid,estimator,model_file_name,models,cm
def Multilayer_Perceptron_Classifier(X_train,y_train,X_test,y_test,grids ,target):
    model_file_name = "Multilayer_Perceptron_Classifier_" + target + '.pkl'
    gd = RandomizedSearchCV(MLPClassifier(), grids, cv=5, n_jobs=-1, verbose=True, refit=True)
    gd.fit(X_train, y_train)
    grid = gd.best_params_
    estimator = gd.best_estimator_
    y_pred = gd.predict(X_test)
    cm1 = confusion_matrix(y_test, y_pred)
    Accuracy = metrics.accuracy_score(y_test, y_pred) * 100
    cm = {'confusion_metrics': cm1.tolist()}
    joblib.dump(estimator, model_file_name)
    perm = PermutationImportance(gd, random_state=1).fit(X_train, y_train)
    importance = perm.feature_importances_
    models = 'Multilayer_Perceptron_Classifier_'
    return Accuracy,y_pred,importance,grid,estimator,model_file_name,models,cm
def SVC_Classifier(X_train,y_train,X_test,y_test,grids ,target):
    model_file_name = "Support_Vector_Classifier_" + target + '.pkl'
    gd = RandomizedSearchCV(SVC(), grids, cv=5, n_jobs=-1, verbose=True, refit=True)
    gd.fit(X_train, y_train)
    grid = gd.best_params_
    estimator = gd.best_estimator_
    y_pred = gd.predict(X_test)
    cm1 = confusion_matrix(y_test, y_pred)
    Accuracy = metrics.accuracy_score(y_test, y_pred) * 100
    cm = {'confusion_metrics': cm1.tolist()}
    joblib.dump(estimator, model_file_name)
    importance = gd.best_estimator_.coef_
    imp = importance.tolist()
    importance = imp[0]
    models = 'Support_Vector_Classifier_'
    return Accuracy,y_pred,importance,grid,estimator,model_file_name,models,cm