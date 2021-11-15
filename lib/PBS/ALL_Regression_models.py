from sklearn.metrics import make_scorer
import os
from QC import train_test
from Fill_DB import DB_upload
from Fill_h2o_DB import h2o_DB_upload
from h2o_Reg import h2o_RF ,h2o_GB ,h2o_XGB ,h2o_Glm
import json
import boto3
import joblib
import numpy as np
import pandas as pd
from math import sqrt
from io import StringIO
from sklearn.impute import SimpleImputer
from smart_open import smart_open 
from os.path import splitext
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.utils import class_weight
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error ,mean_squared_log_error,mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import eli5
from dateutil.parser import parse

class ALL_RGN_Modeling():
    def __init__(self,data_sorted,y,i,types,dname,config_object_user,
                 user_defined_terminology,sample_type ,description ,uom_type,cut_off):
        self.data_sorted = data_sorted
        self.i = i
        self.types = types
        self.dname = dname
        self.config_object_user =config_object_user
        self.user_defined_terminology =user_defined_terminology
        self.sample_type = sample_type
        self.description = description
        self.uom_type =uom_type
        self.cut_off   = cut_off
        self.Regressor = [RandomForestRegressor(),KNeighborsRegressor(),XGBRegressor(),SVR()]
        self.Regressor_grids =[{   
                                'max_depth': [int(x) for x in np.linspace(1, 45, num = 3)],
                                'max_features': ['auto', 'sqrt'],
                                'min_samples_split': [5, 10],
                                'n_estimators': [int(x) for x in np.linspace(start = 20, stop = 200, num = 5)]},
                                {'n_neighbors': np.arange(1, 25)},
                                {'objective':['reg:linear'],
                                  'learning_rate': [0.045], 
                                  'max_depth': [3,4],
                                  'min_child_weight': [2],
                                  'silent': [1],
                                  'subsample': [0.5],         
                                  'n_estimators': [int(x) for x in np.linspace(start = 20, stop = 200, num = 5)]},
                                  {'C' : [0.001, 0.01, 0.1, 1, 10],
                                    'gamma':[0.001, 0.01, 0.1, 1],
                                      'kernel': ['linear']}
                                ]
    def Building_Models_Reg(self):
        dname= self.dname
        user_defined_terminology =self.user_defined_terminology
        sample_type = self.sample_type
        description = self.description
        uom_type =self.uom_type
        config_object_user = self.config_object_user
        cut_off = self.cut_off 
        X = self.data_sorted.drop([self.i],axis=1)
        print(X.shape)
        Y = self.data_sorted[self.i]
        print(Y.unique())
        X= X.fillna(X.mean())
        y = (', '.join(["%s" % self.i]))
        print(y)
        cols = list(self.data_sorted.columns)
        x = cols
        x.remove(y)
        target = self.i
        X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2,random_state = 42)
        X_train, X_test = train_test(X_train, X_test , cut_off)
        models = ['Random_Forest','KNN','XGB','SVR']
        model = "DNN"
        model_file_name = model+ '_' + target+'.h5'
        l =1
        for classifier, params,model in zip(self.Regressor, self.Regressor_grids,models):
            print(classifier)
            model_file_name  =model+"_"+target+'.pkl'
            gd = RandomizedSearchCV(classifier,params,cv = 5, n_jobs=-1,verbose=True,refit = True)
            gd.fit(X_train, y_train)
            y_pred = gd.predict(X_test)
            random_best= gd.best_estimator_.predict(X_test)
            errors = abs(random_best - y_test)
            mape = np.mean(100 * (errors / y_test))
            Accuracy = 100 - mape 
            grid = gd.best_params_
            estimator = gd.best_estimator_
            model_file_name = model+"_"+target+".pkl"
            joblib.dump(estimator, model_file_name)
            print(grid)
            print("Accuracy:",Accuracy)
            print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
            print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
            print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
            cm =None
            target = self.i
            if model=='KNN':
                from eli5.sklearn import PermutationImportance
                perm = PermutationImportance(gd, random_state=1).fit(X_train,y_train)      
                importances = perm.feature_importances_
                feature_list = list(X_train.columns)
                    #create a list of tuples
                feature_importance= sorted(zip(importances, feature_list), reverse=True)                                                #create two lists from the previous list of tuples
                df = pd.DataFrame(feature_importance, columns=['importance', 'feature'])
                print("feature_importances_",df.nlargest(80,'importance'))
                for k,v in grid.items():
                    grid[k] = int(v)
                DB_upload(Accuracy,X_train,X_test,y_test,y_pred,
                                   importances,grid,estimator,l,cm,target,model_file_name,model,dname,config_object_user,
                             user_defined_terminology,sample_type ,description ,uom_type)
            elif model == 'SVR':
                importances = gd.best_estimator_.coef_
                imp = importances.tolist()
                importances = imp[0]
                feature_list = list(X_train.columns)
                #create a list of tuples
                feature_importance= sorted(zip(importances, feature_list), reverse=True)                                                #create two lists from the previous list of tuples
                df = pd.DataFrame(feature_importance, columns=['importance', 'feature'])
                print("feature_importances_",df.nlargest(80,'importance'))
                DB_upload(Accuracy,X_train,X_test,y_test,y_pred, 
                                  importances,grid,estimator,l,cm,target,model_file_name,model,dname,config_object_user,
                             user_defined_terminology,sample_type ,description ,uom_type)
            else:
                importances = gd.best_estimator_.feature_importances_.tolist()
                importances = gd.best_estimator_.feature_importances_
                feature_list = list(X.columns)
                feature_importance= sorted(zip(importances, feature_list), reverse=True)
                #create two lists from the previous list of tuples
                df = pd.DataFrame(feature_importance, columns=['importance', 'feature'])
                importance= list(df['importance'])
                feature= list(df['feature'])
                print(df.nlargest(80,'importance'))
                #create a feature list from the original dataset (list of columns)
                # What are this numbers? Let's get back to the columns of the original dataset
                feature_list = list(X_train.columns)
                #create a list of tuples
                feature_importance= sorted(zip(importances, feature_list), reverse=True)
                for k,v in grid.items():
                    try:
                        if v == int:
                            grid[k] = int(v)
                                   # grid['learning_rate'] == float(0.045)
                    except ValueError:
                        if v== float:
                            grid[key] = float(value)
                importances = importances.astype(float)
                DB_upload(Accuracy,X_train,X_test,y_test,y_pred, 
                              importances,grid,estimator,l,cm,target,model_file_name,model,dname,config_object_user,
                         user_defined_terminology,sample_type ,description ,uom_type)
        def H2o(x,y):
            import h2o
            from h2o.automl import H2OAutoML
            h2o.init()
            X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2,random_state = 42)
            X_train, X_test = train_test(X_train, X_test , cut_off)
            cols = list(X_train.columns)
            cols.append(self.i)
            selected_cols = self.data_sorted[cols]
            x=list(cols)
            df = h2o.H2OFrame(selected_cols)
            train,  test = df.split_frame(ratios=[.8])
            train, test = train_test(train, test , cut_off)
            train[y] = train[y]
            test[y] = test[y]
            # Run AutoML for 20 base models (limited to 1 hour max runtime by default)
            aml = H2OAutoML(max_models=10, seed=10, exclude_algos = ["StackedEnsemble", "DeepLearning"],verbosity="info")
            aml.train(x=x, y=y, training_frame=train)
            # View the AutoML Leaderboard
            model_filename= aml.leader.download_mojo(path = "AI_"+target+"_"+".zip")
            print(model_filename)
            dd = model_filename.split('/')
            model_file_name = (str(dd[2]))
            mojo_model = h2o.import_mojo(model_filename)
            predictions = mojo_model.predict(test)
            lb = aml.leaderboard
            model_ids = list(lb['model_id'].as_data_frame().iloc[:,0])
            moj = model_file_name.replace(".zip", "")
            print(model_ids)
            print(moj)
            for m_id in model_ids:
                mdl = h2o.get_model(m_id)
            rmse = mdl.model_performance().rmse()
            print(rmse)
            cm2 = None
            print(cm2)
            grid = mdl.get_params()
            print(grid)
            mse = mdl.model_performance().mse()
            print(mse)
            r = mdl.model_performance().rmsle()
            print(r)
            Accuracy = mdl.model_performance().mae()
            print(Accuracy)
            Accuracy = None
            importanc = mdl.varimp(use_pandas=True)
            importancs =pd.DataFrame(importanc,columns =['percentage' , 'variable'] )
            print(importancs)
            importances = importancs['percentage']
            h2o_DB_upload(Accuracy,X_train,X_test,rmse,mse,None,importances,
                          grid,None,cm2,y,model_file_name,model,dname,config_object_user,user_defined_terminology,sample_type 
                          ,description ,uom_type)
            print(predictions)
            return lbdf
        H2o(x,y)
        if "Regression_problem" == self.types:
            import keras
            from keras.callbacks import EarlyStopping
            from keras import backend
            import eli5
            from eli5.sklearn import PermutationImportance
            from dateutil.parser import parse
            from scipy.stats import norm
            from keras.models import Sequential
            from keras.layers import Dense
            from keras.wrappers.scikit_learn import KerasRegressor
            import numpy
            ############### to the ai model h2o.save_model(aml.leader, path="./product_backorders_model_bin")
            from keras.models import Sequential
            from keras.layers import Dense, Dropout
            from keras.wrappers.scikit_learn import KerasClassifier
            from keras.layers import BatchNormalization
            from keras.callbacks import EarlyStopping, TensorBoard
            from sklearn.model_selection import cross_val_score, StratifiedKFold
            from keras.constraints import maxnorm
            from keras.layers import Dropout
            def Reg_model():
                model = Sequential()
                model.add(Dense(500, input_dim=X_train.shape[1], activation= "relu"))
                model.add(Dense(100, activation= "relu"))
                model.add(Dense(50, activation= "relu"))
                model.add(Dense(1))
                model.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["accuracy"])
                return model
            model = KerasRegressor(build_fn=Reg_model, verbose=0)
            # define the grid search parameters
            batch_size = [10, 20, 40, 60, 80, 100]
            epochs = [10, 50, 100]
            param_grid = dict(batch_size = batch_size, epochs = epochs)
            grid = GridSearchCV(estimator = model, param_grid=param_grid, n_jobs=-1, cv=3)
            grid_result = grid.fit(X_train, y_train)
            y_pred_keras = grid.predict(X_test)
            estimator = grid.best_estimator_
            Accuracy= grid_result.best_score_
            print(f'Best params: {grid.best_params_}')
            print(f'Best score: {grid.best_score_}')
            print(f'Test MSE: {mean_squared_error(y_true=y_test, y_pred=y_pred_keras)}')
            print(f'Test MAE: {mean_absolute_error(y_true=y_test, y_pred=y_pred_keras)}')
            perm = PermutationImportance(grid, random_state=1).fit(X_train,y_train)
#             eli5.show_weights(perm, feature_names = X.columns.tolist())
            importances=perm.feature_importances_
            feature_list = list(X_train.columns)
            #create a list of tuples
            feature_importance= sorted(zip(importances, feature_list), reverse=True)                                                #create two lists from the previous list of tuples
            df = pd.DataFrame(feature_importance, columns=['importance', 'feature'])
            print("feature_importances_",df.nlargest(80,'importance'))
            estimator.model.save(model_file_name)
            DB_upload(Accuracy,X_train,X_test,y_test,y_pred_keras,importances,
                      grid,estimator,l,None,target,model_file_name,model,dname,config_object_user,user_defined_terminology,sample_type ,description ,uom_type)
            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))