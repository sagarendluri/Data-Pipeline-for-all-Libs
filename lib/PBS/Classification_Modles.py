import os
import sys
import tensorflow as tf
from QC import train_test
from Fill_DB import DB_upload
from h2o_Class import h2o_RF ,h2o_GB ,h2o_XGB ,h2o_Glm
from Fill_h2o_DB import h2o_DB_upload
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
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix ,mean_absolute_error,mean_squared_error
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

class CLF_modeling():
    def __init__(self,data_sorted,y,i,types,algos,sklearn,ai,dname,config_object_user,
                user_defined_terminology,sample_type ,description ,uom_type ,cut_off):
        self.data_sorted = data_sorted
        self.i = i
        self.types = types
        self.algos = algos
        self.sklearn = sklearn
        self.ai =ai
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
    def Classification(self):
#         try:
            algos = self.algos
            types = self.types
            dname= self.dname
            user_defined_terminology =self.user_defined_terminology
            sample_type = self.sample_type
            description = self.description
            uom_type =self.uom_type
            config_object_user = self.config_object_user
            cut_off = self.cut_off
            X = self.data_sorted.drop([self.i],axis=1)
            Y = self.data_sorted[self.i]
            print(Y.unique())
            X= X.fillna(X.mean())
            y = (', '.join(["%s" % self.i]))
            cols = list(self.data_sorted.columns)
            x = cols
            x.remove(y)
#             List of pipelines for ease of iteration
            l = 0
            target = self.i
            model = "DNN"
            models = ['Random_Forest','KNN','XGB','SVC']
            model_file_name = model+"_"+target+'.h5'
            if  'sklearn'== self.sklearn:
                print("models from sklearn")
                def sklearn(X,Y,algos):
                    model = models[algos]
                    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2,random_state = 42)
                    X_train, X_test = train_test(X_train, X_test,cut_off)
                    gd = RandomizedSearchCV(self.Classifier[algos],self.Classifiers_grids[algos],cv = 50, n_jobs=-1,
                                            verbose=True,refit = True)
                    gd.fit(X_train, y_train)
                    grid = gd.best_params_
                    estimator = gd.best_estimator_
                    y_pred=gd.predict(X_test)
                    cm1 =confusion_matrix(y_test, y_pred)
                    
                    Accuracy = metrics.accuracy_score(y_test, y_pred)
                    model_file_name  =model+"_"+target+'.pkl'
                    joblib.dump(estimator, model_file_name)
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
                        feature_importance= sorted(zip(importances, feature_list), reverse=True)  
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
                        feature_importance= sorted(zip(importances, feature_list), reverse=True) 
                        df = pd.DataFrame(feature_importance, columns=['importance', 'feature'])
                        print("feature_importances_",df.nlargest(80,'importance'))
                        DB_upload(Accuracy,X_train,X_test,y_test,y_pred, 
                                      importances,grid,estimator,l,cm,target,model_file_name,model,dname,config_object_user,
                                 user_defined_terminology,sample_type ,description ,uom_type)
                    else:
                        importances = gd.best_estimator_.feature_importances_.tolist()
                        importances = gd.best_estimator_.feature_importances_
                        feature_list = list(X_train.columns)
                        feature_importance= sorted(zip(importances, feature_list), reverse=True)
                        df = pd.DataFrame(feature_importance, columns=['importance', 'feature'])
                        print(df.nlargest(80,'importance'))
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
                    return Accuracy
                sklearn(X,Y,algos)
            elif 'ai' == self.ai:
                print('H2o')
                def H2o(x,y,X,Y):
                    import h2o
                    from h2o.automl import H2OAutoML
                    h2o.init()
                    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2,random_state = 42)
                    X_train, X_test = train_test(X_train, X_test , cut_off)
                    cols = list(X_train.columns)
                    cols.append(self.i)
                    i=self.i
                    selected_cols = self.data_sorted[cols]
                    x=list(cols)
                    df = h2o.H2OFrame(selected_cols)
                    train,  test = df.split_frame(ratios=[.8])
                    train[y] = train[y].asfactor()
                    test[y] = test[y].asfactor()
                    auc,cm,mse,rmse,r2,importances,grid,model_file_name,model=h2o_RF(train,  test,x,y)
                    user_defined_terminology = self.user_defined_terminology+model
                    h2o_DB_upload(auc,X_train,X_test,rmse,mse,r2,importances,
                              grid,None,cm,y,model_file_name,model,dname,config_object_user,user_defined_terminology,sample_type,description ,uom_type)
                    auc,cm,mse,rmse,r2,importances,grid,model_file_name,model=h2o_GB(train,  test,x,y)
                    user_defined_terminology = self.user_defined_terminology+model
                    h2o_DB_upload(auc,X_train,X_test,rmse,mse,r2,importances,
                              grid,None,cm,y,model_file_name,model,dname,config_object_user,user_defined_terminology,sample_type
                              ,description ,uom_type)
                    
                    auc,cm,mse,rmse,r2,importances,grid,model_file_name,model=h2o_XGB(train,  test,x,y)
                    
                    user_defined_terminology = self.user_defined_terminology+model
                    h2o_DB_upload(auc,X_train,X_test,rmse,mse,r2,importances,
                              grid,None,cm,y,model_file_name,model,dname,config_object_user,user_defined_terminology,sample_type,description ,uom_type)
                    auc,cm,mse,rmse,r2,importances,grid,model_file_name,model=h2o_Glm(train,  test,x,y)
                    
                    user_defined_terminology = self.user_defined_terminology+model
                    h2o_DB_upload(auc,X_train,X_test,rmse,mse,r2,importances,
                              grid,None,cm,y,model_file_name,model,dname,config_object_user,user_defined_terminology,sample_type,description ,uom_type)
#                     
                    return lb
                H2o(x,y ,X,Y)
            else:
                    import keras
                    from keras.callbacks import EarlyStopping
                    from keras import backend
                    from keras.wrappers.scikit_learn import KerasClassifier
                    from scipy.stats import norm
                    from keras.models import Sequential
                    from keras.models import Sequential
                    from keras.layers import Dense, Dropout
                    from keras.layers import BatchNormalization
                    from keras.callbacks import EarlyStopping, TensorBoard
                    from sklearn.model_selection import cross_val_score, StratifiedKFold
                    from keras.constraints import maxnorm
                    from keras.layers import Dropout
                    print('Dnn')
                    if types == 'Classification_problem':
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
                        X = self.data_sorted.drop([self.i],axis=1)
                        Y = self.data_sorted[self.i]
                        X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2,random_state = 42)
                        X_train, X_test = train_test(X_train, X_test)
                        classifier = KerasClassifier(build_fn=DNN, verbose=1)
                        batch_size = [10 ,20, 40, 60, 80, 100]
                        epochs = [10, 50, 100]
                        param_grid = dict(batch_size=batch_size, epochs=epochs)
                        grid = GridSearchCV(estimator=classifier, param_grid=param_grid, n_jobs=-1, cv=3)
                        grid_result = grid.fit(X_train, y_train)
                        estimator = grid.best_estimator_
                        Accuracy= grid_result.best_score_
                        estimator.model.save(model_file_name)
#                         model = tf.keras.models.load_model(model_file_name)
#                         y_pred_keras = model.predict(X_test)
                        print("%s" % (estimator))
                        y_pred_keras = grid.predict(X_test)
                        print(f'Best params: {grid.best_params_}')
                        print(f'Best score: {grid.best_score_}')
                        print(f'Test MSE: {mean_squared_error(y_true=y_test, y_pred=y_pred_keras)}')
                        print(f'Test MAE: {mean_absolute_error(y_true=y_test, y_pred=y_pred_keras)}')
                        perm = PermutationImportance(grid, scoring='accuracy', random_state=1).fit(X_train,y_train)      
                        importances=perm.feature_importances_
                        feature_list = list(X_train.columns)
                        feature_importance= sorted(zip(importances, feature_list), reverse=True)
                        df = pd.DataFrame(feature_importance, columns=['importance', 'feature'])
                        print("feature_importances_",df.nlargest(80,'importance'))
                        cm1 =confusion_matrix(y_test, y_pred_keras)
                        cm = {'confusion_metrics':cm1.tolist()}
                        model = "DNN"
                        DB_upload(Accuracy,X_train,X_test,y_test, y_pred_keras,importances,grid,estimator,l,
                                      cm,target,model_file_name,model,dname,config_object_user,user_defined_terminology,sample_type ,description ,uom_type)
                        # summarize results
                        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
                    else:
                        a = np.unique(Y)
                        a.sort()
                        b=a[-1]
                        b +=1
                        print("input_classes",b)
                        def DNN(dropout_rate=0.0, weight_constraint=0):
                            # create model
                            model = Sequential()
                            model.add(Dense(42, input_dim=X_train.shape[1], kernel_initializer='uniform', activation='relu', kernel_constraint=maxnorm(weight_constraint)))
                            model.add(Dropout(dropout_rate))
                            model.add(Dense(20, kernel_initializer='uniform', activation='relu'))
                            model.add(Dense(b,activation='softmax'))
                            model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                            return model
                        X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2,random_state = 42)
                        classifier = KerasClassifier(build_fn=DNN, epochs=10, batch_size=10, verbose=1)
                        weight_constraint = [1, 2, 3, 4, 5]
                        dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                        param_grid = dict(dropout_rate=dropout_rate, weight_constraint=weight_constraint)
                        grid = GridSearchCV(estimator=classifier, param_grid=param_grid, n_jobs=-1, cv=3)
                        grid_result = grid.fit(X_train, y_train)
                        estimator = grid.best_estimator_
                        Accuracy= grid_result.best_score_
                        estimator.model.save(model_file_name)
#                         model = tf.keras.models.load_model(model_file_name)
#                         y_pred_keras = model.predict(X_test)
                        y_pred_keras=grid.predict(X_test)
                        print(y_pred_keras)
                        print("Accuracy",Accuracy)
                        print(f'Best params: {grid.best_params_}')
                        print(f'Best score: {grid.best_score_}')
                        print(f'Test MSE: {mean_squared_error(y_true=y_test, y_pred=y_pred_keras)}')
                        print(f'Test MAE: {mean_absolute_error(y_true=y_test, y_pred=y_pred_keras)}')
                        perm = PermutationImportance(grid, scoring='accuracy', random_state=1).fit(X_train,y_train)      
                        importances=perm.feature_importances_
                        feature_list = list(X_train.columns)
                        feature_importance= sorted(zip(importances, feature_list), reverse=True)
                        df = pd.DataFrame(feature_importance, columns=['importance', 'feature'])
                        print("feature_importances_",df.nlargest(80,'importance'))
                        cm1 =confusion_matrix(y_test, y_pred_keras)
                        print("confusion_metrics",cm1)
                        cm = {'confusion_metrics':cm1.tolist()}
                        DB_upload(Accuracy,X_train,X_test,y_test,y_pred_keras,importances,grid,estimator,l,
                                  cm,target,model_file_name,model,dname,config_object_user,user_defined_terminology,sample_type ,description ,uom_type)

                        print("%s" % (estimator))
#         except:
#             print('Regression model building failed')