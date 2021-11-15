from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from eli5.sklearn import PermutationImportance
def RandomForest_Regressor(X_train,y_train,X_test,y_test,grids ,target,model_):
    gd = RandomizedSearchCV(RandomForestRegressor(), grids, cv=5, n_jobs=-1, verbose=True, refit=True)
    gd.fit(X_train, y_train)
    random_best= gd.best_estimator_.predict(X_test)
    errors = abs(random_best - y_test)
    mape = np.mean(100 * (errors / y_test))
    Accuracy = 100 - mape
    grid =gd.best_params_
    estimator = gd.best_estimator_
    model_file_name = model_+"_Random_Forest_" + target + ".pkl"
    models = 'Random_Forest'
    joblib.dump(estimator, model_file_name)
    importances = gd.best_estimator_.feature_importances_
    for k, v in grid.items():
        try:
            if v == int:
                grid[k] = int(v)
        except ValueError:
            if v == float:
                grid[key] = float(value)
    importances = importances.astype(float)
    return Accuracy,random_best,importances,grid,estimator,model_file_name,models
def XGBoost(X_train,y_train,X_test,y_test,grids,target,model_):
    gd = RandomizedSearchCV(XGBRegressor(), grids, cv=5, n_jobs=-1, verbose=True, refit=True)
    gd.fit(X_train, y_train)
    random_best= gd.best_estimator_.predict(X_test)
    errors = abs(random_best - y_test)
    mape = np.mean(100 * (errors / y_test))
    Accuracy = 100 - mape
    grid =gd.best_params_
    estimator = gd.best_estimator_
    model_file_name = model_+"_XGBoost_" + target + ".pkl"
    models = 'XGBoost'
    joblib.dump(estimator, model_file_name)
    importances = gd.best_estimator_.feature_importances_
    for k, v in grid.items():
        try:
            if v == int:
                grid[k] = int(v)
        except ValueError:
            if v == float:
                grid[key] = float(value)
    importances = importances.astype(float)
    return Accuracy,random_best,importances,grid,estimator,model_file_name,models
def K_Neighbors_Regressor(X_train,y_train,X_test,y_test,grids,target,model_):
    gd = RandomizedSearchCV(KNeighborsRegressor(), grids, cv=5, n_jobs=-1, verbose=True, refit=True)
    gd.fit(X_train, y_train)
    y_pred = gd.predict(X_test)
    random_best = gd.best_estimator_.predict(X_test)
    errors = abs(random_best - y_test)
    mape = np.mean(100 * (errors / y_test))
    Accuracy = 100 - mape
    grid = gd.best_params_
    estimator = gd.best_estimator_
    model_file_name = model_ +"_K_Neighbors_Regressor_" + target + ".pkl"
    models = 'K_Neighbors_Regressor_'
    joblib.dump(estimator, model_file_name)
    cm = "None"
    perm = PermutationImportance(gd, random_state=1).fit(X_train, y_train)
    importances = perm.feature_importances_
    for k, v in grid.items():
        grid[k] = int(v)
    return Accuracy,random_best,importances,grid,estimator,model_file_name,models
def Multilayer_Perceptron_Regressor(X_train,y_train,X_test,y_test,grids,target,model_):
    gd = RandomizedSearchCV(MLPRegressor(), grids, cv=5, n_jobs=-1, verbose=True, refit=True)
    gd.fit(X_train, y_train)
    y_pred = gd.predict(X_test)
    random_best = gd.best_estimator_.predict(X_test)
    errors = abs(random_best - y_test)
    mape = np.mean(100 * (errors / y_test))
    Accuracy = 100 - mape
    grid = gd.best_params_
    estimator = gd.best_estimator_
    model_file_name = model_+"_Multilayer_Perceptron_Regressor_" + target + ".pkl"
    models = 'Multilayer_Perceptron_Regressor_'
    joblib.dump(estimator, model_file_name)
    cm = "None"
    perm = PermutationImportance(gd, random_state=1).fit(X_train, y_train)
    importances = perm.feature_importances_
    return Accuracy,random_best,importances,grid,estimator,model_file_name,models
def Support_vector_regression(X_train,y_train,X_test,y_test,grids,target,model_):
    gd = RandomizedSearchCV(SVR(), grids, cv=5, n_jobs=-1, verbose=True, refit=True)
    gd.fit(X_train, y_train)
    y_pred = gd.predict(X_test)
    random_best = gd.best_estimator_.predict(X_test)
    errors = abs(random_best - y_test)
    mape = np.mean(100 * (errors / y_test))
    Accuracy = 100 - mape
    grid = gd.best_params_
    estimator = gd.best_estimator_
    model_file_name = model_+"_Support_Vector_regression_" + target + ".pkl"
    models = 'Support_Vector_regression_'
    joblib.dump(estimator, model_file_name)
    cm = "None"
    importances = gd.best_estimator_.coef_
    imp = importances.tolist()
    importances = imp[0]
    return Accuracy,random_best,importances,grid,estimator,model_file_name,models

