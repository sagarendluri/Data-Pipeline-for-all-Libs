import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import joblib
from xgboost.sklearn import XGBRegressor
import warnings
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, RFE, RFECV

warnings.filterwarnings('ignore')


def RFECV_RF_Regressor(X_train, y_train, X_test, y_test, grids, target):
    model_file_name = "RandomForest_RFECV_Regressor_" + target + '.pkl'
    rfe = RFECV(estimator=RandomForestRegressor(), cv=5).fit(X_train, y_train)
    gd = RandomizedSearchCV(RandomForestRegressor(), grids, cv=5, n_jobs=-1, verbose=True, refit=True)
    gd.fit(X_train[list(X_train.columns[rfe.support_])], y_train)
    #                 rfe2 = classifier.fit(X_train[list(X_train.columns[gd.support_])], y_train)
    random_best = gd.best_estimator_.predict(X_test[list(X_train.columns[rfe.support_])])
    train_accuracy = gd.score(X_train[list(X_train.columns[rfe.support_])], y_train)
    test_accuracy = gd.score(X_test[list(X_train.columns[rfe.support_])], y_test)
    print("train_accuracy, test_accuracy",train_accuracy, test_accuracy)
    # errors = abs(random_best - y_test)
    # mape = np.mean(100 * (errors / y_test))
    # Accuracy = 100 - mape
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
    models = 'RandomForest_RFECV_Regressor_'
    cm = "None"
    l = 1
    return train_accuracy, test_accuracy, X_train[list(X_train.columns[rfe.support_])], X_test[list(X_train.columns[rfe.support_])], y_test, random_best, importances, grid, estimator, l, cm, target, model_file_name


def RFECV_XGB_Regressor(X_train, y_train, X_test, y_test, grids, target):
    model_file_name = "XGBoost_RFECV_Regressor_" + target + '.pkl'
    rfe = RFECV(estimator=XGBRegressor(), cv=5).fit(X_train, y_train)
    gd = RandomizedSearchCV(XGBRegressor(), grids, cv=5, n_jobs=-1, verbose=True, refit=True)
    gd.fit(X_train[list(X_train.columns[rfe.support_])], y_train)
    #                 rfe2 = classifier.fit(X_train[list(X_train.columns[gd.support_])], y_train)
    random_best = gd.best_estimator_.predict(X_test[list(X_train.columns[rfe.support_])])
    train_accuracy = gd.score(X_train[list(X_train.columns[rfe.support_])], y_train)
    test_accuracy = gd.score(X_test[list(X_train.columns[rfe.support_])], y_test)
    print("train_accuracy, test_accuracy", train_accuracy, test_accuracy)
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
    models = 'XGBoost_RFECV_Regressor_'
    cm = "None"
    l = 1
    return train_accuracy, test_accuracy, X_train[list(X_train.columns[rfe.support_])], X_test[list(X_train.columns[rfe.support_])], y_test, random_best, importances, grid, estimator, l, cm, target, model_file_name
