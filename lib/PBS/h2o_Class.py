from h2o.estimators import H2ORandomForestEstimator
from h2o.estimators import H2OGradientBoostingEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators import H2OXGBoostEstimator
import pandas as pd
def h2o_RF(train, test ,predictors, response):
    model = H2ORandomForestEstimator(ntrees=10,
                                        max_depth=5,
                                        min_rows=10)
    model.train(x=predictors, y=response, training_frame=train)
    performance= model.model_performance(test_data=test)
    print(performance)
    auc = performance.auc()
    cm = performance.confusion_matrix().as_data_frame()
    cm = cm.to_dict()
    mse = performance.mse()
    r2 = performance.r2()
    rmse = performance.rmse()
    importanc = model.varimp(use_pandas=True)
    importancs =pd.DataFrame(importanc,columns =['percentage' , 'variable'] )
    print(importancs)
    importances = importancs['percentage']
    grid = model.get_params()
    print(grid)
    model_file_name = "H2ORandomForest_"+response+"_"+".zip"
    model.download_mojo(model_file_name)
    models = "H2ORandomForestEstimator"
    return auc,cm,mse,rmse,r2,importances,grid,model_file_name,models
def h2o_GB(train, test ,predictors, response):
    pros_gbm = H2OGradientBoostingEstimator(nfolds=5,
                                            seed=1111,
                                            keep_cross_validation_predictions = True)
    pros_gbm.train(x=predictors, y=response, training_frame=train)
    performance = pros_gbm.model_performance()
    print(performance)
    auc = performance.auc()
    cm = performance.confusion_matrix().as_data_frame()
    cm = cm.to_dict()
    mse = performance.mse()
    r2 = performance.r2()
    rmse = performance.rmse()
    importanc = pros_gbm.varimp(use_pandas=True)
    importancs =pd.DataFrame(importanc,columns =['percentage' , 'variable'] )
    print(importancs)
    importances = importancs['percentage']
    grid = pros_gbm.get_params()  
    print(grid)
    model_file_name = "H2OGradientBoosting_"+response+"_"+".zip"
    pros_gbm.download_mojo(model_file_name)
    model = "H2OGradientBoostingEstimator"
    return auc,cm,mse,rmse,r2,importances,grid,model_file_name,model
def h2o_XGB(train, test ,predictors, response):
    model = H2OXGBoostEstimator(booster='dart',
                                  normalize_type="tree",
                                  seed=1234)
    model.train(x=predictors, y=response, training_frame=train)
    performance = model.model_performance(test_data=test)
    print(performance)
    auc = performance.auc()
    cm = performance.confusion_matrix().as_data_frame()
    cm = cm.to_dict()
    mse = performance.mse()
    r2 = performance.r2()
    rmse = performance.rmse()
    importanc = model.varimp(use_pandas=True)
    importancs =pd.DataFrame(importanc,columns =['percentage' , 'variable'] )
    print(importancs)
    importances = importancs['percentage']
    grid = model.get_params()  
    print(grid)
    model_file_name = "H2OXGBoostEstimator_"+response+"_"+".zip"
    model.download_mojo(model_file_name)
    model = "H2OXGBoostEstimator"
    return auc,cm,mse,rmse,r2,importances,grid,model_file_name,model
def h2o_Glm(train, test ,predictors, response):
    model = H2OGeneralizedLinearEstimator(family= "multinomial",
                                          lambda_ = 0)
    model.train(x=predictors, y=response, training_frame=train)
    print(model.coef())
    performance = model.model_performance(test_data=test)
    print(performance)
    auc = performance.auc()
    cm = performance.confusion_matrix().as_data_frame()
    cm = cm.to_dict()
    mse = performance.mse()
    r2 = performance.r2()
    rmse = performance.rmse()
    importanc = model.varimp(use_pandas=True)
    importancs =pd.DataFrame(importanc,columns =['percentage' , 'variable'] )
    print(importancs)
    importances = importancs['percentage']
    grid = model.get_params()  
    print(grid)
    model_file_name = "H2OGeneralizedLinearEstimator_"+response+"_"+".zip"
    model.download_mojo(model_file_name)
    model = "H2OGeneralizedLinearEstimator"
    return auc,cm,mse,rmse,r2,importances,grid,model_file_name,model