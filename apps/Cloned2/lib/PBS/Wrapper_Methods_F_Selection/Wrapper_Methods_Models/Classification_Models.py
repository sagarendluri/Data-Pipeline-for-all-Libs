import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import joblib
from xgboost.sklearn import XGBClassifier
import warnings
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, RFE, RFECV

warnings.filterwarnings('ignore')


def RFECV_RF_Classifier(X_train, y_train, X_test, y_test, grids, target):
    model_file_name = "RandomForest_RFECV_Classifier_" + target + '.pkl'
    rfe = RFECV(estimator=RandomForestClassifier(), cv=5).fit(X_train, y_train)
    gd = RandomizedSearchCV(RandomForestClassifier(), grids, cv=5, n_jobs=-1, verbose=True, refit=True)
    gd.fit(X_train[list(X_train.columns[rfe.support_])], y_train)
    #                 rfe2 = classifier.fit(X_train[list(X_train.columns[gd.support_])], y_train)
    random_best = gd.best_estimator_.predict(X_test[list(X_train.columns[rfe.support_])])
    train_accuracy = gd.score(X_train, y_train)
    test_accuracy = gd.score(X_test, y_test)
    clsf_report = pd.DataFrame(
        classification_report(y_true=y_test, y_pred=random_best, output_dict=True)).transpose()
    print(clsf_report)
    grid = gd.best_params_
    estimator = gd.best_estimator_
    joblib.dump(estimator, model_file_name)
    importances = gd.best_estimator_.feature_importances_
    for k, v in grid.items():
        try:
            if v == int:
                grid[k] = int(v)
                # grid['learning_rate'] == float(0.045)
        except ValueError:
            if v == float:
                grid[key] = float(value)
    importances = importances.astype(float)
    models = 'RandomForest_RFECV_Classifier_'
    cm = "None"
    l = 1
    metrics.classification_report(y_test, y_pred_class)
    return train_accuracy, test_accuracy, X_train[list(X_train.columns[rfe.support_])], X_test[list(X_train.columns[
                                                                                                        rfe.support_])], y_test, random_best, importances, grid, estimator, l, cm, target, model_file_name


def RFECV_XGB_Classifier(X_train, y_train, X_test, y_test, grids, target):
    model_file_name = "XGBoost_RFECV_Classifier_" + target + '.pkl'
    rfe = RFECV(estimator=XGBClassifier(), cv=5).fit(X_train, y_train)
    gd = RandomizedSearchCV(XGBClassifier(), grids, cv=5, n_jobs=-1, verbose=True, refit=True)
    gd.fit(X_train[list(X_train.columns[rfe.support_])], y_train)
    #                 rfe2 = classifier.fit(X_train[list(X_train.columns[gd.support_])], y_train)
    random_best = gd.best_estimator_.predict(X_test[list(X_train.columns[rfe.support_])])
    train_accuracy = gd.score(X_train, y_train)
    test_accuracy = gd.score(X_test, y_test)
    grid = gd.best_params_
    estimator = gd.best_estimator_
    clsf_report = pd.DataFrame(
        classification_report(y_true=y_test, y_pred=random_best, output_dict=True)).transpose()
    print(clsf_report)
    joblib.dump(estimator, model_file_name)
    importances = gd.best_estimator_.feature_importances_
    for k, v in grid.items():
        try:
            if v == int:
                grid[k] = int(v)
                # grid['learning_rate'] == float(0.045)
        except ValueError:
            if v == float:
                grid[key] = float(value)
    importances = importances.astype(float)
    models = 'XGBoost_RFECV_Classifier_'
    cm = "None"
    l = 1
    return train_accuracy, test_accuracy, X_train[list(X_train.columns[rfe.support_])], X_test[list(X_train.columns[
                                                                                                        rfe.support_])], y_test, random_best, importances, grid, estimator, l, cm, target, model_file_name
