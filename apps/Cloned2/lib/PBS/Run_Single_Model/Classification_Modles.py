import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import accuracy_score ,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier
import eli5
from eli5.sklearn import PermutationImportance
from dateutil.parser import parse
from sklearn.neural_network import MLPClassifier
from apps.Cloned2.lib.PBS.Data_preprocessing.QC import train_test
from apps.Cloned2.lib.PBS.Upload_Data_Database.Fill_DB import DB_upload
from apps.Cloned2.lib.PBS.Upload_Data_Database.Fill_h2o_DB import h2o_DB_upload
import apps.Cloned2.lib.PBS.Upload_Data_Database.DB_details
from apps.Cloned2.lib.PBS.H2O.h2o_Class import h2o_RF, h2o_GB, h2o_XGB, h2o_Glm
class CLF_modeling:
    def __init__(self, data_sorted, y, i, types, algos, sklearn, ai, dname, config_object_user,
                 user_defined_terminology, sample_type, description, uom_type, cut_off, analysis_id, min_depth,
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
        self.Classifier = [RandomForestClassifier(), KNeighborsClassifier(), XGBClassifier(), SVC(), MLPClassifier()]
    def parameter_tuning(self):
        if "default" == self.default:
            self.Classifiers_grids = [{
                'max_depth': [int(x) for x in np.linspace(1, 45, num=3)],
                'max_features': ['auto', 'sqrt'],
                'min_samples_split': [5, 10],
                'n_estimators': [int(x) for x in np.linspace(start=20, stop=200, num=5)]},
                {'n_neighbors': np.arange(1, 25)},
                {'objective': ['reg:linear'],
                 'learning_rate': [0.045],
                 'max_depth': [3, 4],
                 'min_child_weight': [2],
                 'n_estimators': [int(x) for x in np.linspace(start=20, stop=200, num=5)]},
                {'C': [0.001, 0.01, 0.1, 1, 10],
                 'gamma': [0.001, 0.01, 0.1, 1],
                 'kernel': ['linear']},
                {'learning_rate_init': [0.0001],
                 'max_iter': [300],
                 'hidden_layer_sizes': [(500, 400, 300, 200, 100),
                                        (400, 400, 400, 400, 400),
                                        (300, 300, 300, 300, 300),
                                        (200, 200, 200, 200, 200)],
                 'activation': ['logistic', 'relu', 'tanh'],
                 'alpha': [0.0001, 0.001, 0.005],
                 'early_stopping': [True, False]}
            ]
            return self.Classifiers_grids
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
                 'kernel': self.svm_kernel},
                {'learning_rate_init': [0.0001],
                 'max_iter': [300],
                 'hidden_layer_sizes': [(500, 400, 300, 200, 100),
                                        (400, 400, 400, 400, 400),
                                        (300, 300, 300, 300, 300),
                                        (200, 200, 200, 200, 200)],
                 'activation': ['logistic', 'relu', 'tanh'],
                 'alpha': [0.0001, 0.001, 0.005],
                 'early_stopping': [True, False]}
            ]
            return self.Classifiers_grids

    def Classification(self):
        # try:
            algos = self.algos
            types = self.types
            dname = self.dname
            db_name =  self.db_name
            user_defined_terminology = self.user_defined_terminology
            sample_type = self.sample_type
            description = self.description
            uom_type = self.uom_type
            config_object_user = self.config_object_user
            cut_off = self.cut_off
            analysis_id = self.analysis_id
            X = self.data_sorted.drop([self.i], axis=1)
            Y = self.data_sorted[self.i]
            X = X.fillna(X.mean())
            cols = list(self.data_sorted.columns)
            y = (', '.join(["%s" % self.i]))
            x = cols
            x.remove(y)
            l = 0
            target = self.i
            model = "DNN"
            models = ['Random_Forest', 'KNN', 'XGB', 'SVC', 'MLP_NN']
            model_file_name = model + "_" + target + '.h5'
            if 'sklearn' == self.sklearn:

                def sklearn(X, Y, algos):
                    model = models[algos]
                    print(self.Classifier[algos])
                    print(self.Classifiers_grids[algos])
                    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
                    X_train, X_test = train_test(X_train, X_test, cut_off)
                    gd = RandomizedSearchCV(self.Classifier[algos], self.Classifiers_grids[algos], cv=5, n_jobs=-1,
                                            verbose=True, refit=True)
                    gd.fit(X_train, y_train)
                    grid = gd.best_params_
                    estimator = gd.best_estimator_
                    y_pred = gd.predict(X_test)
                    cm1 = confusion_matrix(y_test, y_pred)
                    Accuracy = metrics.accuracy_score(y_test, y_pred)
                    model_file_name = model + "_" + target + '.pkl'
                    joblib.dump(estimator, model_file_name)
                    cm = {'confusion_metrics': cm1.tolist()}
                    if model == 'KNN':
                        perm = PermutationImportance(gd, random_state=1).fit(X_train, y_train)
                        importances = perm.feature_importances_
                        print("feature_importances_", df.nlargest(80, 'importance'))
                        for k, v in grid.items():
                            grid[k] = int(v)
                        DB_upload(Accuracy, X_train, X_test, y_test, y_pred,
                                  importances, grid, estimator, l, cm, target, model_file_name, model, dname,
                                  config_object_user,
                                  user_defined_terminology, sample_type, description, uom_type, analysis_id,db_name)
                    elif model == 'MLP_NN':
                        perm = PermutationImportance(gd, random_state=1).fit(X_train, y_train)
                        importance = perm.feature_importances_  # create two lists from the previous list of tuples
                        DB_upload(Accuracy, X_train, X_test, y_test, y_pred,
                                  importance, grid, estimator, l, cm, target, model_file_name, model, dname,
                                  config_object_user,
                                  user_defined_terminology, sample_type, description, uom_type, analysis_id,db_name)
                    elif model == 'SVC':
                        importances = gd.best_estimator_.coef_
                        imp = importances.tolist()
                        importances = imp[0]
                        DB_upload(Accuracy, X_train, X_test, y_test, y_pred,
                                  importances, grid, estimator, l, cm, target, model_file_name, model, dname,
                                  config_object_user,
                                  user_defined_terminology, sample_type, description, uom_type, analysis_id,db_name)
                    else:
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
                        DB_upload(Accuracy, X_train, X_test, y_test, y_pred,
                                  importances, grid, estimator, l, cm, target, model_file_name, model, dname,
                                  config_object_user,
                                  user_defined_terminology, sample_type, description, uom_type, analysis_id,db_name)
                    return Accuracy

                sklearn(X, Y, algos)
            elif 'ai' == self.ai:
                def H2o(x, y, X, Y):
                    import h2o
                    from h2o.automl import H2OAutoML
                    h2o.init()
                    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
                    X_train, X_test = train_test(X_train, X_test, cut_off)
                    cols = list(X_train.columns)
                    cols.append(self.i)
                    i = self.i
                    selected_cols = self.data_sorted[cols]
                    x = list(cols)
                    X_train[y] = y_test
                    X_test[y] = y_test
                    df = h2o.H2OFrame(selected_cols)
                    X_test = h2o.H2OFrame(X_test[list(X_train.columns)])
                    train, test = df.split_frame(ratios=[.8])
                    train[y] = train[y].asfactor()
                    test[y] = test[y].asfactor()
                    db_train = X_train.drop(y, axis=1)
                    db_test = X_test.drop(y, axis=1)
                    auc, cm, mse, rmse, r2, importances, grid, model_file_name, model = h2o_RF(train, X_test, x, y,
                                                                                               y_test)
                    user_defined_terminology = self.user_defined_terminology
                    h2o_DB_upload(auc, db_train, db_test, rmse, mse, r2, importances, grid, "None", cm, y,
                                  model_file_name, model, dname, config_object_user, user_defined_terminology,
                                  sample_type, description, uom_type, analysis_id,db_name)
                    auc, cm, mse, rmse, r2, importances, grid, model_file_name, model = h2o_GB(train, X_test, x, y,
                                                                                               y_test)
                    user_defined_terminology = self.user_defined_terminology
                    h2o_DB_upload(auc, db_train, db_test, rmse, mse, r2, importances,
                                  grid, "None", cm, y, model_file_name, model, dname, config_object_user,
                                  user_defined_terminology, sample_type
                                  , description, uom_type, analysis_id,db_name)
                    auc, cm, mse, rmse, r2, importances, grid, model_file_name, model = h2o_XGB(train, X_test, x, y,
                                                                                                y_test)

                    user_defined_terminology = self.user_defined_terminology
                    h2o_DB_upload(auc, db_train, db_test, rmse, mse, r2, importances,
                                  grid, "None", cm, y, model_file_name, model, dname, config_object_user,
                                  user_defined_terminology, sample_type, description, uom_type, analysis_id,db_name)
                    auc, cm, mse, rmse, r2, importances, grid, model_file_name, model = h2o_Glm(train, X_test, x, y,
                                                                                                y_test)

                    user_defined_terminology = self.user_defined_terminology
                    h2o_DB_upload(auc, db_train, db_test, rmse, mse, r2, importances,
                                  grid, "None", cm, y, model_file_name, model, dname, config_object_user,
                                  user_defined_terminology, sample_type, description, uom_type, analysis_id,db_name)

                    return "loaded data into db"

                H2o(x, y, X, Y)
            else:
                import keras
                from keras.callbacks import EarlyStopping
                from keras import backend
                from keras.wrappers.scikit_learn import KerasClassifier
                from scipy.stats import norm
                from keras.models import Sequential
                ############## to the ai model h2o.save_model(aml.leader, path="./product_backorders_model_bin")
                from keras.models import Sequential
                from keras.layers import Dense, Dropout
                from keras.layers import BatchNormalization
                from keras.callbacks import EarlyStopping, TensorBoard
                from sklearn.model_selection import cross_val_score, StratifiedKFold
                from keras.constraints import maxnorm
                from keras.layers import Dropout
                if types == 'Classification_problem':
                    model_file_name = "DNN_" + target + '.h5'
                    a = np.unique(Y)
                    a.sort()
                    b = a[-1]
                    b += 1

                    def DNN():
                        model = Sequential()
                        model.add(Dense(52, input_dim=X_train.shape[1], init='normal', activation='relu'))
                        model.add(BatchNormalization())
                        model.add(Dropout(0.5))
                        model.add(Dense(32, init='normal', activation='relu'))
                        model.add(BatchNormalization())
                        model.add(Dropout(0.5))
                        model.add(Dense(b, init='normal', activation='softmax'))
                        model.compile(loss='sparse_categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
                        return model

                    classifier = KerasClassifier(build_fn=DNN, verbose=1)
                    batch_size = [10, 20, 40, 60, 80, 100]
                    epochs = [10, 50, 100]
                    param_grid = dict(batch_size=batch_size, epochs=epochs)
                    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
                    X_train, X_test = train_test(X_train, X_test, cut_off)
                    grid = GridSearchCV(estimator=classifier, param_grid=param_grid, n_jobs=-1, cv=3)
                    grid_result = grid.fit(X_train, y_train)
                    estimator = grid.best_estimator_
                    y_pred_keras = grid.predict(X_test)
                    cm1 = confusion_matrix(y_test, y_pred_keras)
                    target = self.i
                    Accuracy = metrics.accuracy_score(y_test, y_pred_keras)
                    cm = {'confusion_metrics': cm1.tolist()}
                    grid.best_estimator_.model.save(model_file_name)
                    grids = grid.best_params_
                    perm = PermutationImportance(grid, scoring='accuracy', random_state=1).fit(X_train, y_train)
                    importance = perm.feature_importances_
                    DB_upload(Accuracy, X_train, X_test, y_test, y_pred_keras, importance, grids, estimator, l,
                              cm, target, model_file_name, model, dname, config_object_user, user_defined_terminology,
                              sample_type, description, uom_type, analysis_id, db_name)
                else:
                    a = np.unique(self.y)
                    a.sort()
                    b = a[-1]
                    b += 1

                    def DNN(dropout_rate=0.0, weight_constraint=0):
                        # create model
                        model = Sequential()
                        model.add(Dense(42, input_dim=X_train.shape[1], kernel_initializer='uniform', activation='relu',
                                        kernel_constraint=maxnorm(weight_constraint)))
                        model.add(Dropout(dropout_rate))
                        model.add(Dense(20, kernel_initializer='uniform', activation='relu'))
                        model.add(Dense(b, activation='softmax'))
                        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                        return model

                    classifier = KerasClassifier(build_fn=DNN, epochs=50, batch_size=10, verbose=1)
                    weight_constraint = [1, 2, 3, 4, 5]
                    dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                    param_grid = dict(dropout_rate=dropout_rate, weight_constraint=weight_constraint)
                    grid = GridSearchCV(estimator=classifier, param_grid=param_grid, n_jobs=-1, cv=3)
                    grid_result = grid.fit(X_train, y_train)
                    estimator = grid.best_estimator_
                    y_pred_keras = grid.predict(X_test)
                    cm1 = confusion_matrix(y_test, y_pred_keras)
                    target = self.i
                    Accuracy = metrics.accuracy_score(y_test, y_pred_keras)
                    cm = {'confusion_metrics': cm1.tolist()}
                    grid.best_estimator_.model.save(model_file_name)
                    grids = grid.best_params_
                    perm = PermutationImportance(grid, scoring='accuracy', random_state=1).fit(X_train, y_train)
                    importance = perm.feature_importances_
                    DB_upload(Accuracy, X_train, X_test, y_test, y_pred_keras, importance, grids, estimator, l,
                              cm, target, model_file_name, model, dname, config_object_user, user_defined_terminology,
                              sample_type, description, uom_type, analysis_id, db_name)
                #         except:
                #             print('Regression model building failed')
                # except Exception as Ex:
                #     print("ALL_Classification_Models exited with the error : ")  # $%" % (Ex))
                #     print(Ex)
