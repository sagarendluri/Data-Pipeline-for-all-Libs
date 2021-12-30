from h2o.estimators import H2ORandomForestEstimator
from h2o.estimators import H2OGradientBoostingEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators import H2OXGBoostEstimator

from h2o.estimators import H2ODeepLearningEstimator
import pandas as pd
import numpy as np
from sklearn import metrics
import math


def h2o_RF(train, test, predictors, response, y_test, analysis_id):
    model = H2ORandomForestEstimator(ntrees=20,
                                     max_depth=5,
                                     min_rows=5)
    model.train(x=predictors, y=response, training_frame=train)  # ,validation_frame=test)
    train_accuracy = model.r2()
    performance = model.model_performance(test_data=test)
    y_pred = model.predict(test).as_data_frame(use_pandas=True)
    y_pred = y_pred['predict'].to_list()
    y_pred = grid.best_estimator_.predict(X_test)
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    y_test = y_test.reshape(len(y_test), 1)
    y_pred = y_pred.reshape(len(y_pred), 1)
    diff = (y_test - y_pred)
    mbe = diff.mean()
    cm = "Nan"
    mse = performance.mse()
    r2 = performance.r2()
    r2 = r2 * 100
    test_accuracy = r2
    rmse = performance.rmse()
    importanc = model.varimp(use_pandas=True)
    importancs = pd.DataFrame(importanc, columns=['percentage', 'variable'])
    variable_order = importancs['variable']
    importances = importancs['percentage']
    grid = model.get_params()
    print(grid)
    model_file_name = "H2ORandomForest_" + "_" + analysis_id + "_" + response + "_" + ".zip"
    model.download_mojo(model_file_name)
    models = "H2ORandomForestEstimator"
    return train_accuracy, test_accuracy, cm, mse, rmse, r2, importances, grid, model_file_name, models, variable_order,y_pred ,mbe


def h2o_GB(train, test, predictors, response, y_test, analysis_id):
    pros_gbm = H2OGradientBoostingEstimator(nfolds=6,
                                            seed=1111,
                                            keep_cross_validation_predictions=True)
    pros_gbm.train(x=predictors, y=response, training_frame=train)
    r2 = pros_gbm.r2() * 100
    train_accuracy = r2
    performance = pros_gbm.model_performance(test_data=test)
    y_pred = pros_gbm.predict(test).as_data_frame(use_pandas=True)
    y_pred = y_pred['predict'].to_list()
    y_pred = grid.best_estimator_.predict(X_test)
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    y_test = y_test.reshape(len(y_test), 1)
    y_pred = y_pred.reshape(len(y_pred), 1)
    diff = (y_test - y_pred)
    mbe = diff.mean()
    cm = "Nan"
    mse = performance.mse()
    r2 = performance.r2()
    r2 = r2 * 100
    test_accuracy = r2
    rmse = performance.rmse()
    importanc = pros_gbm.varimp(use_pandas=True)
    importancs = pd.DataFrame(importanc, columns=['percentage', 'variable'])
    variable_order = importancs['variable']
    importances = importancs['percentage']
    grid = pros_gbm.get_params()
    model_file_name = "H2OGradientBoosting_" + "_" + analysis_id + response + "_" + ".zip"
    pros_gbm.download_mojo(model_file_name)
    models = "H2OGradientBoostingEstimator"
    return train_accuracy, test_accuracy, cm, mse, rmse, r2, importances, grid, model_file_name, models, variable_order ,y_pred ,mbe


def h2o_XGB(train, test, predictors, response, y_test, analysis_id):
    model = H2OXGBoostEstimator(booster='dart',
                                normalize_type="tree",
                                seed=1234)
    model.train(x=predictors, y=response, training_frame=train)
    r2 = model.r2() * 100
    train_accuracy = r2
    performance = model.model_performance(test_data=test)
    y_pred = model.predict(test).as_data_frame(use_pandas=True)
    y_pred = y_pred['predict'].to_list()
    y_pred = grid.best_estimator_.predict(X_test)
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    y_test = y_test.reshape(len(y_test), 1)
    y_pred = y_pred.reshape(len(y_pred), 1)
    diff = (y_test - y_pred)
    mbe = diff.mean()
    print("y_test", y_test)
    print("y_pred", y_pred)
    cm = "Nan"
    mse = performance.mse()
    r2 = performance.r2()
    r2 = r2 * 100
    test_accuracy = r2
    rmse = performance.rmse()
    importanc = model.varimp(use_pandas=True)
    importancs = pd.DataFrame(importanc, columns=['percentage', 'variable'])
    variable_order = importancs['variable']
    importances = importancs['percentage']
    grid = model.get_params()
    model_file_name = "H2OXGBoostEstimator_" + analysis_id + "_" + response + "_" + ".zip"
    model.download_mojo(model_file_name)
    models = "H2OXGBoostEstimator"
    return train_accuracy, test_accuracy, cm, mse, rmse, r2, importances, grid, model_file_name, models, variable_order ,y_pred ,mbe


def h2o_Glm(train, test, predictors, response, y_test, analysis_id):
    model = H2OGeneralizedLinearEstimator(family="gaussian",
                                          lambda_=0,
                                          compute_p_values=True)
    model.train(predictors, response, training_frame=train)
    r2 = model.r2() * 100
    train_accuracy = r2
    performance = model.model_performance(test_data=test)
    y_pred = model.predict(test).as_data_frame(use_pandas=True)
    y_pred = y_pred['predict'].to_list()
    y_pred = grid.best_estimator_.predict(X_test)
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    y_test = y_test.reshape(len(y_test), 1)
    y_pred = y_pred.reshape(len(y_pred), 1)
    diff = (y_test - y_pred)
    mbe = diff.mean()
    print(y_test)
    print(y_pred['predict'])
    cm = "Nan"
    mse = performance.mse()
    r2 = performance.r2()
    r2 = r2 * 100
    test_accuracy = r2
    rmse = performance.rmse()
    importanc = model.varimp(use_pandas=True)
    importancs = pd.DataFrame(importanc, columns=['percentage', 'variable'])
    variable_order = importancs['variable']
    importances = importancs['percentage']
    grid = model.get_params()
    model_file_name = "H2OGeneralizedLinearEstimator_" + analysis_id + "_" + response + "_" + ".zip"
    model.download_mojo(model_file_name)
    models = "H2OGeneralizedLinearEstimator"
    return train_accuracy, test_accuracy, cm, mse, rmse, r2, importances, grid, model_file_name, models, variable_order,y_pred , mbe


def h2o_DeepNN(train, test, predictors, response, y_test, analysis_id):
    dl = H2ODeepLearningEstimator(distribution="gaussian",
                                  hidden=[1],
                                  epochs=1000,
                                  train_samples_per_iteration=-1,
                                  reproducible=True,
                                  activation="Tanh",
                                  single_node_mode=False,
                                  balance_classes=False,
                                  force_load_balance=False,
                                  seed=23123,
                                  tweedie_power=1.5,
                                  score_training_samples=0,
                                  score_validation_samples=0,
                                  stopping_rounds=0)
    dl.train(x=predictors, y=response, training_frame=train)
    r2 = dl.r2() * 100
    train_accuracy = r2
    performance = dl.model_performance(test_data=test)
    y_pred = dl.predict(test).as_data_frame(use_pandas=True)
    y_pred = y_pred['predict'].to_list()
    y_pred = grid.best_estimator_.predict(X_test)
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    y_test = y_test.reshape(len(y_test), 1)
    y_pred = y_pred.reshape(len(y_pred), 1)
    diff = (y_test - y_pred)
    mbe = diff.mean()
    mse = performance.mse()
    r2 = performance.r2()
    r2 = r2 * 100
    test_accuracy = r2
    cm = "Nan"
    rmse = performance.rmse()
    importanc = dl.varimp(use_pandas=True)
    importancs = pd.DataFrame(importanc, columns=['percentage', 'variable'])
    print(importancs)
    variable_order = importancs['variable']
    importances = importancs['percentage']
    grid = dl.get_params()
    print(grid)
    model_file_name = "H2ODeepNN_" + analysis_id + "_" + response + "_" + ".zip"
    dl.download_mojo(model_file_name)
    models = "H2ODeepNN"
    return train_accuracy, test_accuracy, cm, mse, rmse, r2, importances, grid, model_file_name, models, variable_order,y_pred ,mbe
