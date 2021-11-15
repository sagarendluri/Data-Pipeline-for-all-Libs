import os
import sys
from QC import train_test
from Fill_DB import DB_upload
from Fill_h2o_DB import h2o_DB_upload

from h2o_Class import h2o_RF ,h2o_GB ,h2o_XGB ,h2o_Glm
import DB_details
import json

import joblib
import argparse
import numpy as np
import pandas as pd
from math import sqrt
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import accuracy_score,mean_absolute_error
from sklearn.metrics import confusion_matrix,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV 
from sklearn.model_selection import GridSearchCV 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.svm import SVC
import eli5
from eli5.sklearn import PermutationImportance
from dateutil.parser import parse


class sk_Models():
    def __init__(self,data_sorted, y, i, types, dname,config_object_user,
                                 user_defined_terminology,sample_type ,description ,uom_type,cut_off):
        self.data_sorted = data_sorted
        self.i = i
        self.types = types
        self.dname = dname
        self.config_object_user= config_object_user
        self.user_defined_terminology =user_defined_terminology
        self.sample_type = sample_type
        self.description = description
        self.uom_type =uom_type
        self.cut_off = cut_off
        self.Classifier = [RandomForestClassifier(),KNeighborsClassifier(),XGBClassifier(),SVC()]
        self.Classifiers_grids = [{   
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
                                      'kernel': ['linear']}]
    def Building_Models(self):
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
            X_train, X_test = train_test(X_train, X_test ,cut_off)
            models = ['Random_Forest','KNN','XGB','SVC']
            model = "DNN"
            l =1
            for classifier, params,model in zip(self.Classifier, self.Classifiers_grids,models):
                print(classifier)
                model_file_name  =model+"_"+target+'.pkl'
                joblib.dump(estimator, model_file_name)
                gd = RandomizedSearchCV(classifier,params,cv = 5, n_jobs=-1,verbose=True,refit = True)
                gd.fit(X_train, y_train)
                grid = gd.best_params_
                estimator = gd.best_estimator_
                y_pred=gd.predict(X_test)
                cm1 =confusion_matrix(y_test, y_pred)
                target = self.i
                Accuracy = metrics.accuracy_score(y_test, y_pred)
                cm = {'confusion_metrics':cm1.tolist()}
                print("CM",cm)
                print("GRIDS",grid)
                print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
                print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
                print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
                print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
                if model=='KNN':
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
                elif model == 'SVC':
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
            def H2o(x,y,X,Y):
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
                train[y] = train[y].asfactor()
                test[y] = test[y].asfactor()
                # Run AutoML for 20 base models (limited to 1 hour max runtime by default)
                aml = H2OAutoML(max_models=5, seed=10, exclude_algos = ["StackedEnsemble", "DeepLearning"], verbosity="info")
                aml.train(x=x, y=y, training_frame=train)
                # View the AutoML Leaderboard
                model_filename= aml.leader.download_mojo(path = "AI_"+target+"_"+".zip")
                print(model_filename)
                dd = model_filename.split('/')
                model_file_name = (str(dd[2]))
                mojo_model = h2o.import_mojo(model_filename)
                predictions = mojo_model.predict(test)
                print(predictions)
                lb = aml.leaderboard
                model_ids = list(lb['model_id'].as_data_frame().iloc[:,0])
                moj = model_file_name.replace(".zip", "")
                print(model_ids)
                print(moj)
                for m_id in model_ids:
                    mdl = h2o.get_model(m_id)
                rmse = mdl.model_performance().rmse()
                print(rmse)
                cm = mdl.model_performance().confusion_matrix().as_data_frame()
                cm2 = dict(cm)
                print(cm2)
                grid = mdl.get_params()
                print(grid)
                mse = mdl.model_performance().mse()
                print(mse)
                r = mdl.model_performance().r2()
                print(r)
                Accuracy = mdl.model_performance().auc()
                print(Accuracy)
                importanc = mdl.varimp(use_pandas=True)
                importancs =pd.DataFrame(importanc,columns =['percentage' , 'variable'] )
                print(importancs)
                importances = importancs['percentage']
                h2o_DB_upload(Accuracy,X_train,X_test,rmse,mse,None,importances,
                              grid,None,cm2,y,model_file_name,model,dname,config_object_user,user_defined_terminology,sample_type
                              ,description ,uom_type)
                return lb
            H2o(x,y ,X,Y)
            if "Classification_problem" == 'Classification_problem':
                import keras
                from keras.callbacks import EarlyStopping
                from keras import backend
                from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
                from scipy.stats import norm
                from keras.models import Sequential
                from keras.layers import Dense
                from keras.wrappers.scikit_learn import KerasRegressor
                ############## to the ai model h2o.save_model(aml.leader, path="./product_backorders_model_bin")
                from keras.models import Sequential
                from keras.layers import Dense, Dropout
                from keras.wrappers.scikit_learn import KerasClassifier
                from keras.layers import BatchNormalization
                from keras.callbacks import EarlyStopping, TensorBoard
                from sklearn.model_selection import cross_val_score, StratifiedKFold
                from keras.constraints import maxnorm
                from keras.layers import Dropout
                model_file_name = model+"_"+target+'.h5'
                def DNN():
                    model = Sequential()
                    model.add(Dense(512, input_dim=X_train.shape[1], init='normal', activation='relu'))
                    model.add(BatchNormalization())
                    model.add(Dropout(0.5))
                    model.add(Dense(32, init='normal', activation='relu'))
                    model.add(BatchNormalization())
                    model.add(Dropout(0.5))
                    model.add(Dense(1, init='normal', activation='sigmoid'))
                    model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
                    return model
                classifier = KerasClassifier(build_fn=DNN, verbose=1)
                batch_size = [10 ,20, 40, 60, 80, 100]
                epochs = [10, 50, 100]
                param_grid = dict(batch_size=batch_size, epochs=epochs)
                grid = GridSearchCV(estimator=classifier, param_grid=param_grid, n_jobs=-1, cv=3)
                grid_result = grid.fit(X_train, y_train)
                estimator = grid.best_estimator_
                print("%s" % (estimator))
                y_pred_keras=grid.predict(X_test)
                print(y_pred_keras)
                cm1 =confusion_matrix(y_test, y_pred_keras)
                target = self.i
                Accuracy = metrics.accuracy_score(y_test, y_pred_keras)
                cm = {'confusion_metrics':cm1.tolist()}
                print("CM",cm)
                print("Accuracy",Accuracy)
                print(f'Best params: {grid.best_params_}')
                print(f'Best score: {grid.best_score_}')
                print(f'Test MSE: {mean_squared_error(y_true=y_test, y_pred=y_pred_keras)}')
                print(f'Test MAE: {mean_absolute_error(y_true=y_test, y_pred=y_pred_keras)}')
                perm = PermutationImportance(grid, scoring='accuracy', random_state=1).fit(X_train,y_train)      
                importances= perm.feature_importances_
                feature_list = list(X_train.columns)
                #create a list of tuples
                feature_importance= sorted(zip(importances, feature_list), reverse=True)                                                #create two lists from the previous list of tuples
                df = pd.DataFrame(feature_importance, columns=['importance', 'feature'])
                print("feature_importances_",df.nlargest(80,'importance'))
                DB_upload(Accuracy,X_train,X_test,y_test, y_pred_keras,importances,grid,estimator,l,
                                      cm,target,model_file_name,model,dname,config_object_user,user_defined_terminology,sample_type ,description ,uom_type)
            else:
                a = np.unique(self.y)
                a.sort()
                b=a[-1]
                b +=1
                def DNN(dropout_rate=0.0, weight_constraint=0):
                    # create model
                    model = Sequential()
                    model.add(Dense(42, input_dim=X_train.shape[1], kernel_initializer='uniform', activation='relu', kernel_constraint=maxnorm(weight_constraint)))
                    model.add(Dropout(dropout_rate))
                    model.add(Dense(20, kernel_initializer='uniform', activation='relu'))
                    model.add(Dense(b,activation='softmax'))
                    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                    return model
                classifier = KerasClassifier(build_fn=DNN, epochs=50, batch_size=10, verbose=1)
                weight_constraint = [1, 2, 3, 4, 5]
                dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                param_grid = dict(dropout_rate=dropout_rate, weight_constraint=weight_constraint)
                grid = GridSearchCV(estimator=classifier, param_grid=param_grid, n_jobs=-1, cv=3)
                grid_result = grid.fit(X_train, y_train)
                estimator = grid.best_estimator_
                print("%s" % (estimator))
                y_pred_keras=grid.predict(X_test)
                print(y_pred_keras)
                cm1 =confusion_matrix(y_test, y_pred_keras)
                target = self.i
                Accuracy = metrics.accuracy_score(y_test, y_pred_keras)
                cm = {'confusion_metrics':cm1.tolist()}
                print("CM",cm)
                print("Accuracy",Accuracy)
                print(f'Best params: {grid.best_params_}')
                print(f'Best score: {grid.best_score_}')
                print(f'Test MSE: {mean_squared_error(y_true=y_test, y_pred=y_pred_keras)}')
                print(f'Test MAE: {mean_absolute_error(y_true=y_test, y_pred=y_pred_keras)}')
                perm = PermutationImportance(grid, scoring='accuracy', random_state=1).fit(X_train,y_train)      
                importances= perm.feature_importances_
                feature_list = list(X_train.columns)
                #create a list of tuples
                feature_importance= sorted(zip(importances, feature_list), reverse=True)                                                #create two lists from the previous list of tuples
                df = pd.DataFrame(feature_importance, columns=['importance', 'feature'])
                print("feature_importances_",df.nlargest(80,'importance'))
                DB_upload(Accuracy,X_train,X_test,y_test, y_pred_keras,importances,grid,estimator,l,
                                      cm,target,model_file_name,model,dname,config_object_user,user_defined_terminology,sample_type ,description ,uom_type)
        #         except:
        #             print('Regression model building failed')