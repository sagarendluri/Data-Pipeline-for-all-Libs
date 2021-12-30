from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from eli5.sklearn import PermutationImportance


def RandomForest_Regressor(X_train, y_train, X_test, y_test, grids, target, model_):
    rfg = RandomizedSearchCV(RandomForestRegressor(), grids, cv=5, n_jobs=-1, verbose=True, refit=True)
    rfg.fit(X_train, y_train)
    train_accuracy = rfg.score(X_train, y_train)*100
    test_accuracy = rfg.score(X_test, y_test)*100
    random_best = rfg.best_estimator_.predict(X_test)
    print("train_accuracy,test_accuracy", train_accuracy, test_accuracy)
    y_pred = rfg.best_estimator_.predict(X_test)
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    y_test = y_test.reshape(len(y_test), 1)
    y_pred = y_pred.reshape(len(y_pred), 1)
    diff = (y_test - y_pred)
    mbe = diff.mean()
    # errors = abs(random_best - y_test)
    # mape = np.mean(100 * (errors / y_test))
    # Accuracy = 100 - mape
    grid = rfg.best_params_
    estimator = rfg.best_estimator_
    model_file_name = model_ + "_Random_Forest_" + target + ".pkl"
    models = 'Random_Forest'
    joblib.dump(estimator, model_file_name)
    importances = rfg.best_estimator_.feature_importances_
    for k, v in grid.items():
        try:
            if v == int:
                grid[k] = int(v)
        except ValueError:
            if v == float:
                grid[key] = float(value)
    importances = importances.astype(float)
    return train_accuracy, test_accuracy, random_best, importances, grid, estimator, model_file_name, models ,mbe


def XGBoost(X_train, y_train, X_test, y_test, grids, target, model_):
    xgb = RandomizedSearchCV(XGBRegressor(), grids, cv=5, n_jobs=-1, verbose=True, refit=True)
    xgb.fit(X_train, y_train)
    train_accuracy = xgb.score(X_train, y_train)*100
    test_accuracy = xgb.score(X_test, y_test)*100
    print("train_accuracy,test_accuracy", train_accuracy, test_accuracy)
    random_best = xgb.best_estimator_.predict(X_test)
    y_pred = xgb.best_estimator_.predict(X_test)
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    y_test = y_test.reshape(len(y_test), 1)
    y_pred = y_pred.reshape(len(y_pred), 1)
    diff = (y_test - y_pred)
    mbe = diff.mean()
    # errors = abs(random_best - y_test)
    # mape = np.mean(100 * (errors / y_test))
    # Accuracy = 100 - mape
    grid = xgb.best_params_
    estimator = xgb.best_estimator_
    model_file_name = model_ + "_XGBoost_" + target + ".pkl"
    models = 'XGBoost'
    joblib.dump(estimator, model_file_name)
    importances = xgb.best_estimator_.feature_importances_
    for k, v in grid.items():
        try:
            if v == int:
                grid[k] = int(v)
        except ValueError:
            if v == float:
                grid[key] = float(value)
    importances = importances.astype(float)
    return train_accuracy, test_accuracy, random_best, importances, grid, estimator, model_file_name, models ,mbe


def K_Neighbors_Regressor(X_train, y_train, X_test, y_test, grids, target, model_):
    knn = RandomizedSearchCV(KNeighborsRegressor(), grids, cv=5, n_jobs=-1, verbose=True, refit=True)
    knn.fit(X_train, y_train)
    train_accuracy = knn.score(X_train, y_train)*100
    test_accuracy = knn.score(X_test, y_test)*100
    print("train_accuracy,test_accuracy", train_accuracy, test_accuracy)
    random_best = knn.best_estimator_.predict(X_test)
    y_pred = knn.best_estimator_.predict(X_test)
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    y_test = y_test.reshape(len(y_test), 1)
    y_pred = y_pred.reshape(len(y_pred), 1)
    diff = (y_test - y_pred)
    mbe = diff.mean()
    # errors = abs(random_best - y_test)
    # mape = np.mean(100 * (errors / y_test))
    # Accuracy = 100 - mape
    grid = knn.best_params_
    estimator = knn.best_estimator_
    model_file_name = model_ + "_K_Neighbors_Regressor_" + target + ".pkl"
    models = 'K_Neighbors_Regressor_'
    joblib.dump(estimator, model_file_name)
    cm = "None"
    perm = PermutationImportance(knn, random_state=1).fit(X_train, y_train)
    importances = perm.feature_importances_
    for k, v in grid.items():
        grid[k] = int(v)
    return train_accuracy, test_accuracy, random_best, importances, grid, estimator, model_file_name, models,mbe


def Multilayer_Perceptron_Regressor(X_train, y_train, X_test, y_test, grids, target, model_):
    mlp = RandomizedSearchCV(MLPRegressor(), grids, cv=5, n_jobs=-1, verbose=True, refit=True)
    mlp.fit(X_train, y_train)
    train_accuracy = mlp.score(X_train, y_train)*100
    test_accuracy = mlp.score(X_test, y_test)*100
    print("train_accuracy,test_accuracy", train_accuracy, test_accuracy)
    random_best = mlp.best_estimator_.predict(X_test)
    y_pred = mlp.best_estimator_.predict(X_test)
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    y_test = y_test.reshape(len(y_test), 1)
    y_pred = y_pred.reshape(len(y_pred), 1)
    diff = (y_test - y_pred)
    mbe = diff.mean()
    # errors = abs(random_best - y_test)
    # mape = np.mean(100 * (errors / y_test))
    # Accuracy = 100 - mape
    grid = mlp.best_params_
    estimator = mlp.best_estimator_
    model_file_name = model_ + "_Multilayer_Perceptron_Regressor_" + target + ".pkl"
    models = 'Multilayer_Perceptron_Regressor_'
    joblib.dump(estimator, model_file_name)
    cm = "None"
    perm = PermutationImportance(mlp, random_state=1).fit(X_train, y_train)
    importances = perm.feature_importances_
    return train_accuracy, test_accuracy, random_best, importances, grid, estimator, model_file_name, models ,mbe


def Support_vector_regression(X_train, y_train, X_test, y_test, grids, target, model_):
    svm = RandomizedSearchCV(SVR(), grids, cv=5, n_jobs=-1, verbose=True, refit=True)
    svm.fit(X_train, y_train)
    train_accuracy = svm.score(X_train, y_train)*100
    test_accuracy = svm.score(X_test, y_test)*100
    print("train_accuracy,test_accuracy", train_accuracy, test_accuracy)
    random_best = svm.best_estimator_.predict(X_test)
    y_pred = svm.best_estimator_.predict(X_test)
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    y_test = y_test.reshape(len(y_test), 1)
    y_pred = y_pred.reshape(len(y_pred), 1)
    diff = (y_test - y_pred)
    mbe = diff.mean()
    # errors = abs(random_best - y_test)
    # mape = np.mean(100 * (errors / y_test))
    # Accuracy = 100 - mape
    grid = svm.best_params_
    estimator = svm.best_estimator_
    model_file_name = model_ + "_Support_Vector_regression_" + target + ".pkl"
    models = 'Support_Vector_regression_'
    joblib.dump(estimator, model_file_name)
    cm = "None"
    importances = svm.best_estimator_.coef_
    imp = importances.tolist()
    importances = imp[0]
    return train_accuracy, test_accuracy, random_best, importances, grid, estimator, model_file_name, models ,mbe
