import os
import joblib
import numpy as np
import pandas as pd
from math import sqrt
from smart_open import smart_open
from sklearn import metrics
import eli5
from eli5.sklearn import PermutationImportance
from dateutil.parser import parse
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error
from apps.Cloned2.lib.PBS.Data_preprocessing.QC import train_test
from apps.Cloned2.lib.PBS.Upload_Data_Database.Fill_DB import DB_upload
from apps.Cloned2.lib.PBS.Upload_Data_Database.Fill_h2o_DB import h2o_DB_upload
import apps.Cloned2.lib.PBS.Upload_Data_Database.DB_details
from apps.Cloned2.lib.PBS.H2O.h2o_Reg import h2o_RF, h2o_GB, h2o_XGB, h2o_Glm
from sklearn.ensemble import RandomForestRegressor



class RGN_Modeling:
    def __init__(self, data_sorted, y, i, types, algos, sklearn, ai, dname, config_object_user,
                 user_defined_terminology, sample_type, description, uom_type, cut_off, analysis_id,
                 min_depth,
                 max_depth,
                 min_samples_split,
                 n_estimators_start,
                 n_estimators_stop,
                 n_neighbors,
                 xgb_objective,
                 xgb_learning_rate,
                 xgb_max_depth,
                 xgb_min_child_weight,
                 svm_c,
                 svm_gamma,
                 svm_kernel,
                 default,
                 db_name):
        self.data_sorted = data_sorted
        self.i = i
        self.types = types
        self.algos = algos
        self.sklearn = sklearn
        self.ai = ai
        self.dname = dname
        self.config_object_user = config_object_user
        self.user_defined_terminology = user_defined_terminology
        self.sample_type = sample_type
        self.description = description
        self.uom_type = uom_type
        self.cut_off = cut_off
        self.analysis_id = analysis_id
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_estimators_start = n_estimators_start
        self.n_estimators_stop = n_estimators_stop
        self.n_neighbors = n_neighbors
        self.xgb_objective = xgb_objective
        self.xgb_learning_rate = xgb_learning_rate
        self.xgb_max_depth = xgb_max_depth
        self.xgb_min_child_weight = xgb_min_child_weight
        self.svm_c = svm_c
        self.svm_gamma = svm_gamma
        self.svm_kernel = svm_kernel
        self.default = default
        self.db_name = db_name
        self.Regressor = [RandomForestRegressor(), KNeighborsRegressor(), XGBRegressor(), SVR(),MLPRegressor()]

    def parameter_tuning(self):
        if "default" == self.default:
            self.Regressor_grids = [{
                'max_depth': [int(x) for x in np.linspace(1, 45, num=3)],
                'max_features': ['auto', 'sqrt'],
                'min_samples_split': [5, 10],
                'n_estimators': [int(x) for x in np.linspace(start=20, stop=200, num=5)]},
                {'n_neighbors': np.arange(1, 25)},
                {'objective': ['reg:linear'],
                 'learning_rate': [0.045],
                 'max_depth': [3, 4],
                 'min_child_weight': [2],
                 'silent': [1],
                 'subsample': [0.5],
                 'n_estimators': [int(x) for x in np.linspace(start=20, stop=200, num=5)]},
                {'C': [0.001, 0.01, 0.1, 1, 10],
                 'gamma': [0.001, 0.01, 0.1, 1],
                 'kernel': ['linear'],
                 },{'learning_rate_init': [0.0001],
                 'max_iter': [300],
                 'hidden_layer_sizes': [(500, 400, 300, 200, 100),
                                        (400, 400, 400, 400, 400),
                                        (300, 300, 300, 300, 300),
                                        (200, 200, 200, 200, 200)],
                 'activation': ['logistic', 'relu', 'tanh'],
                 'alpha': [0.0001, 0.001, 0.005],
                 'early_stopping': [True, False]},
            ]
            return self.Regressor_grids
        else:
            import ast
            self.Classifiers_grids = [{
                'max_depth': [int(x) for x in np.linspace([ast.literal_eval(x) for x in self.min_depth][0],
                                                          [ast.literal_eval(x) for x in self.max_depth][0], num=5)],
                'max_features': ['auto', 'sqrt'],
                'min_samples_split': [ast.literal_eval(x) for x in self.self.min_samples_split],

                'n_estimators': [int(x) for x in
                                 np.linspace(start=[ast.literal_eval(x) for x in self.n_estimators_start][0],
                                             stop=[ast.literal_eval(x) for x in self.n_estimators_stop][0], num=5)]},

                {'n_neighbors': np.arange(1, [ast.literal_eval(x) for x in self.n_neighbors][0])},
                {
                    'learning_rate': [ast.literal_eval(x) for x in self.xgb_learning_rate],
                    'max_depth': [ast.literal_eval(x) for x in self.xgb_max_depth],
                    'min_child_weight': [ast.literal_eval(x) for x in self.xgb_min_child_weight],
                    'n_estimators': [int(x) for x in np.linspace(start=
                                                                 [ast.literal_eval(x) for x in self.n_estimators_start][
                                                                     0],
                                                                 stop=
                                                                 [ast.literal_eval(x) for x in self.n_estimators_stop][
                                                                     0],
                                                                 num=5)]},
                {'C': [ast.literal_eval(x) for x in self.svm_c],
                 'gamma': [ast.literal_eval(x) for x in self.svm_gamma],
                 'kernel': self.svm_kernel}]
            return self.Classifiers_grids

    def Regression(self):
        # try:
            algos = self.algos
            types = self.types
            dname = self.dname
            user_defined_terminology = self.user_defined_terminology
            sample_type = self.sample_type
            description = self.description
            uom_type = self.uom_type
            cut_off = self.cut_off
            analysis_id = self.analysis_id
            config_object_user = self.config_object_user
            db_name = self.db_name
            X = self.data_sorted.drop([self.i], axis=1)
            Y = self.data_sorted[self.i]
            Modles_reuslts = []
            Names = []
            target = self.i
            models = ['Random_Forest', 'KNN', 'XGB', 'SVR',"MLP_NN"]
            l = 0
            features = []
            X = X.fillna(X.mean())
            y = (', '.join(["%s" % self.i]))
            cols = list(self.data_sorted.columns)
            x = cols
            x.remove(y)
            model = 'DNN'
            model_file_name = model + '_' + target + '.h5'
            if 'sklearn' == self.sklearn:
                def sklearn(X, Y, algos):
                    model = models[algos]
                    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
                    X_train, X_test = train_test(X_train, X_test, cut_off)
                    gd = RandomizedSearchCV(self.Regressor[algos], self.Regressor_grids[algos], cv=5, n_jobs=-1,
                                            verbose=True, refit=True)
                    gd.fit(X_train, y_train)
                    y_pred = gd.predict(X_test)
                    random_best = gd.best_estimator_.predict(X_test)
                    errors = abs(random_best - y_test)
                    mape = np.mean(100 * (errors / y_test))
                    Accuracy = 100 - mape
                    grid = gd.best_params_
                    estimator = gd.best_estimator_
                    model_file_name = model + "_" + target + ".pkl"
                    joblib.dump(estimator, model_file_name)
                    #                     print("Accuracy:",Accuracy)
                    #                     print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
                    #                     print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
                    #                     print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
                    if model == 'KNN':
                        perm = PermutationImportance(gd, random_state=1).fit(X_train, y_train)
                        importances = perm.feature_importances_
                        feature_list = list(X_train.columns)
                        feature_importance = sorted(zip(importances, feature_list), reverse=True)
                        df = pd.DataFrame(feature_importance, columns=['importance', 'feature'])
                        for k, v in grid.items():
                            grid[k] = int(v)
                        DB_upload(Accuracy, X_train, X_test, y_test, y_pred,
                                  importances,
                                  grid, estimator, l, "None", target, model_file_name, model, dname, config_object_user,
                                  user_defined_terminology, sample_type, description, uom_type, analysis_id,db_name)
                    elif model == 'SVR':
                        weights = gd.best_estimator_.coef_
                        imp = weights.tolist()
                        importances = imp[0]
                        feature_list = list(X_train.columns)
                        feature_importance = sorted(zip(importances, feature_list), reverse=True)
                        df = pd.DataFrame(feature_importance, columns=['importance', 'feature'])
                        DB_upload(Accuracy, X_train, X_test, y_test,
                                  y_pred, importances,
                                  grid, estimator, l, "None", target, model_file_name, model, dname, config_object_user,
                                  user_defined_terminology, sample_type, description, uom_type, analysis_id,db_name)
                    elif model == 'MLP_NN':
                        perm = PermutationImportance(gd, random_state=1).fit(X_train, y_train)
                        importances = perm.feature_importances_
                        feature_list = list(X_train.columns)
                        feature_importance = sorted(zip(importances, feature_list),
                                                    reverse=True)  # create two lists from the previous list of tuples
                        df = pd.DataFrame(feature_importance, columns=['importance', 'feature'])
                        print(grid)
                        # for k, v in grid.items():
                        #     grid[k] = int(v)
                        DB_upload(Accuracy, X_train, X_test, y_test, y_pred,
                                  importances, grid, estimator, l, "None", target, model_file_name, model, dname,
                                  config_object_user,
                                  user_defined_terminology, sample_type, description, uom_type, analysis_id, db_name)
                    else:
                        importances = gd.best_estimator_.feature_importances_
                        feature_list = list(X_train.columns)
                        feature_importance = sorted(zip(importances, feature_list), reverse=True)
                        df = pd.DataFrame(feature_importance, columns=['importance', 'feature'])
                        features.append(importances)
                        for k, v in grid.items():
                            try:
                                if v == int:
                                    grid[k] = int(v)
                                    # grid['learning_rate'] == float(0.045)
                            except ValueError:
                                if v == float:
                                    grid[key] = float(value)
                        importances = importances.astype(float)
                        DB_upload(Accuracy, X_train, X_test, y_test, y_pred, importances,
                                  grid, estimator, l, "None", target, model_file_name, model, dname, config_object_user,
                                  user_defined_terminology, sample_type, description, uom_type, analysis_id,db_name)
                        feature_list = list(X_train.columns)
                        feature_importance = sorted(zip(importances, feature_list), reverse=True)
                    return Accuracy

                sklearn(X, Y, algos)
            elif 'ai' == self.ai:
                def H2o(x, y):
                    import h2o
                    h2o.init()
                    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
                    X_train, X_test = train_test(X_train, X_test, cut_off)
                    cols = list(X_train.columns)
                    cols.append(self.i)
                    selected_cols = self.data_sorted[cols]
                    x = list(cols)
                    df = h2o.H2OFrame(selected_cols)
                    X_train[y] = y_test
                    X_test[y] = y_test
                    X_test = h2o.H2OFrame(X_test[list(X_train.columns)])
                    train, test = df.split_frame(ratios=[.8])
                    train[y] = train[y]
                    test[y] = test[y]
                    auc, cm, mse, rmse, r2, importances, grid, model_file_name, model = h2o_RF(train, X_test, x, y,
                                                                                               y_test)
                    user_defined_terminology = self.user_defined_terminology
                    db_train = X_train.drop(y, axis=1)
                    db_test = X_test.drop(y, axis=1)
                    h2o_DB_upload(auc, db_train, db_test, rmse, mse, r2, importances,
                                  grid, " ", cm, y, model_file_name, model, dname, config_object_user,
                                  user_defined_terminology, sample_type, description, uom_type, analysis_id,db_name)
                    auc, cm, mse, rmse, r2, importances, grid, model_file_name, model = h2o_GB(train, X_test, x, y,
                                                                                               y_test)
                    user_defined_terminology = self.user_defined_terminology
                    h2o_DB_upload(auc, db_train, db_test, rmse, mse, r2, importances,
                                  grid, " ", cm, y, model_file_name, model, dname, config_object_user,
                                  user_defined_terminology, sample_type
                                  , description, uom_type, analysis_id,db_name)
                    auc, cm, mse, rmse, r2, importances, grid, model_file_name, model = h2o_XGB(train, X_test, x, y,
                                                                                                y_test)
                    user_defined_terminology = self.user_defined_terminology
                    h2o_DB_upload(auc, db_train, db_test, rmse, mse, r2, importances,
                                  grid, " ", cm, y, model_file_name, model, dname, config_object_user,
                                  user_defined_terminology, sample_type, description, uom_type, analysis_id)
                    auc, cm, mse, rmse, r2, importances, grid, model_file_name, model = h2o_Glm(train, X_test, x, y,
                                                                                                y_test)
                    user_defined_terminology = self.user_defined_terminology
                    h2o_DB_upload(auc, db_train, db_test, rmse, mse, r2, importances,
                                  grid, " ", cm, y, model_file_name, model, dname, config_object_user,
                                  user_defined_terminology, sample_type, description, uom_type, analysis_id,db_name)
                    return "good"

                H2o(x, y)
            else:
                import keras
                from keras.callbacks import EarlyStopping
                from keras import backend
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
                    model.add(Dense(500, input_dim=X_train.shape[1], activation="relu"))
                    model.add(Dense(100, activation="relu"))
                    model.add(Dense(50, activation="relu"))
                    model.add(Dense(1))
                    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
                    return model

                model = KerasRegressor(build_fn=Reg_model, verbose=0)
                # define the grid search parameters
                batch_size = [10]
                epochs = [10]
                param_grid = dict(batch_size=batch_size, epochs=epochs)
                X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
                X_train, X_test = train_test(X_train, X_test, cut_off)
                grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
                grid_result = grid.fit(X_train, y_train)
                y_pred_keras = grid.predict(X_test)
                estimator = grid.best_estimator_
                Accuracy = grid_result.best_score_
                grids = grid.best_params_
                model_name = "Deep_Neural_Nets"
                #                 print(f'Best params: {grid.best_params_}')
                #                 print(f'Best score: {grid.best_score_}')
                #                 print(f'Test MSE: {mean_squared_error(y_true=y_test, y_pred=y_pred_keras)}')
                #                 print(f'Test MAE: {mean_absolute_error(y_true=y_test, y_pred=y_pred_keras)}')
                perm = PermutationImportance(grid, random_state=1).fit(X_train, y_train)
                #             eli5.show_weights(perm, feature_names = X.columns.tolist())
                importances = perm.feature_importances_
                estimator.model.save(model_file_name)
                DB_upload(Accuracy, X_train, X_test, y_test, y_pred_keras, importances,
                          grids, estimator, l, "None", target, model_file_name, model_name, dname, config_object_user,
                          user_defined_terminology, sample_type, description, uom_type ,analysis_id,db_name)
        #                 print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        # except Exception as Ex:
        #     print("Regression_Models exited with the error : ")  # $%" % (Ex))
        #     print(Ex)
