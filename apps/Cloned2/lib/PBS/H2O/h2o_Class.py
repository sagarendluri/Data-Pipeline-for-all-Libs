from h2o.estimators import H2ORandomForestEstimator
from h2o.estimators import H2OGradientBoostingEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators import H2OXGBoostEstimator
import pandas as pd
import numpy as np
from sklearn import metrics
from h2o.estimators import H2ODeepLearningEstimator


def h2o_RF(train, test, predictors, response, y_test,dname):
    model = H2ORandomForestEstimator(ntrees=10,
                                     max_depth=5,
                                     min_rows=10)
    model.train(x=predictors, y=response, training_frame=train)
    performance = model.model_performance(test_data=test)
    y_pred = model.predict(test).as_data_frame(use_pandas=True)
    Accuracy = metrics.accuracy_score(y_test, y_pred.predict) * 100
    print("ACC", Accuracy)
    print(performance)
    #     auc = performance.auc()
    cm = performance.confusion_matrix().as_data_frame()
    cm = cm.to_dict()
    mse = performance.mse()
    r2 = performance.r2()
    r2 = r2 * 100
    rmse = performance.rmse()
    importanc = model.varimp(use_pandas=True)
    importancs = pd.DataFrame(importanc, columns=['percentage', 'variable'])
    print(importancs)
    variable_order = importancs['variable']
    importances = importancs['percentage']
    grid = model.get_params()
    print(grid)
    model_file_name = "H2ORandomForest_" +dname+ "_"+ response + "_" + ".zip"
    model.download_mojo(model_file_name)
    models = "H2ORandomForestEstimator"
    return r2, cm, mse, rmse, r2, importances, grid, model_file_name, models,variable_order


def h2o_GB(train, test, predictors, response, y_test,dname):
    pros_gbm = H2OGradientBoostingEstimator(nfolds=5,
                                            seed=1111,
                                            keep_cross_validation_predictions=True)
    pros_gbm.train(x=predictors, y=response, training_frame=train)
    performance = pros_gbm.model_performance(test_data=test)
    print(performance)
    y_pred = pros_gbm.predict(test).as_data_frame(use_pandas=True)
    Accuracy = metrics.accuracy_score(y_test, y_pred.predict) * 100
    print("ACC", Accuracy)
    #     auc = performance.auc()
    cm = performance.confusion_matrix().as_data_frame()
    cm = cm.to_dict()
    mse = performance.mse()
    r2 = performance.r2()
    r2 = r2 * 100
    rmse = performance.rmse()
    importanc = pros_gbm.varimp(use_pandas=True)
    print("h2o",importanc)
    importancs = pd.DataFrame(importanc, columns=['percentage', 'variable'])
    print("importancs_df",importancs)
    variable_order = importancs['variable']
    importances = importancs['percentage']
    print("importances",importances)
    grid = pros_gbm.get_params()
    print(grid)
    model_file_name = "H2OGradientBoosting_" + dname+ "_" + response + "_" + ".zip"
    pros_gbm.download_mojo(model_file_name)
    models = "H2OGradientBoostingEstimator"
    return r2, cm, mse, rmse, r2, importances, grid, model_file_name, models,variable_order
def h2o_XGB(train, test, predictors, response, y_test,dname):
    model = H2OXGBoostEstimator(booster='dart',
                                normalize_type="tree",
                                seed=1234)
    model.train(x=predictors, y=response, training_frame=train)
    performance = model.model_performance(test_data=test)
    y_pred = model.predict(test).as_data_frame(use_pandas=True)
    Accuracy = metrics.accuracy_score(y_test, y_pred.predict) * 100
    print("ACC", Accuracy)
    #     auc = performance.auc()
    cm = performance.confusion_matrix().as_data_frame()
    cm = cm.to_dict()
    mse = performance.mse()
    r2 = performance.r2()
    r2 = r2 * 100
    rmse = performance.rmse()
    importanc = model.varimp(use_pandas=True)
    importancs = pd.DataFrame(importanc, columns=['percentage', 'variable'])
    print("importancs_df",importancs)
    variable_order = importancs['variable']
    importances = importancs['percentage']
    grid = model.get_params()
    print(grid)
    model_file_name = "H2OXGBoostEstimator_" +dname+ "_" + response + "_" + ".zip"
    model.download_mojo(model_file_name)
    models = "H2OXGBoostEstimator"
    return r2, cm, mse, rmse, r2, importances, grid, model_file_name, models,variable_order
def h2o_Glm(train, test, predictors, response, y_test,dname):
    model = H2OGeneralizedLinearEstimator(family="multinomial",
                                          lambda_=0)
    model.train(x=predictors, y=response, training_frame=train)
    print(model.coef())
    performance = model.model_performance(test_data=test)
    y_pred = model.predict(test).as_data_frame(use_pandas=True)
    Accuracy = metrics.accuracy_score(y_test, y_pred.predict) * 100
    print("ACC", Accuracy)
    cm = performance.confusion_matrix().as_data_frame()
    cm = cm.to_dict()
    mse = performance.mse()
    r2 = performance.r2()
    r2 = r2 * 100
    rmse = performance.rmse()
    importanc = model.varimp(use_pandas=True)
    importancs = pd.DataFrame(importanc, columns=['percentage', 'variable'])
    print("importancs_df",importancs)
    variable_order = importancs['variable']
    importances = importancs['percentage']
    grid = model.get_params()
    print(grid)
    model_file_name = "H2OGeneralizedLinearEstimator_" +dname + "_" + response + "_" + ".zip"
    model.download_mojo(model_file_name)
    models = "H2OGeneralizedLinearEstimator"
    return r2, cm, mse, rmse, r2, importances, grid, model_file_name, models,variable_order


def h2o_DeepNN(train, test, predictors, response, y_test,dname):
    dl = H2ODeepLearningEstimator(distribution="multinomial",
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
    performance = dl.model_performance(test_data=test)
    y_pred = dl.predict(test).as_data_frame(use_pandas=True)
    Accuracy = metrics.accuracy_score(y_test, y_pred.predict) * 100
    print("ACC", Accuracy)
    cm = performance.confusion_matrix().as_data_frame()
    cm = cm.to_dict()
    mse = performance.mse()
    r2 = performance.r2()
    r2 = r2 * 100
    rmse = performance.rmse()
    importanc = dl.varimp(use_pandas=True)
    importancs = pd.DataFrame(importanc, columns=['percentage', 'variable'])
    print("importancs_df",importancs)
    variable_order = importancs['variable']
    importances = importancs['percentage']
    grid = dl.get_params()
    print(grid)
    model_file_name = "H2ODeepNN_" + dname + "_" + response + "_" + ".zip"
    dl.download_mojo(model_file_name)
    models = "H2ODeepNN"
    return r2, cm, mse, rmse, r2, importances, grid, model_file_name, models,variable_order
