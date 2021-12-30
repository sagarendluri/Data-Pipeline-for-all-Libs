import os
import pandas as pd
import boto3
from apps.Cloned2.lib.PBS.Data_preprocessing.QC import train_test
from apps.Cloned2.lib.PBS.Upload_Data_Database.Fill_DB import DB_upload
from apps.Cloned2.lib.PBS.Upload_Data_Database.Fill_h2o_DB import h2o_DB_upload
import apps.Cloned2.lib.PBS.Upload_Data_Database.DB_details
from apps.Cloned2.lib.PBS.H2O.h2o_Reg import h2o_RF, h2o_GB, h2o_XGB, h2o_Glm, h2o_DeepNN
from apps.Cloned2.lib.PBS.L1_and_L2.l1_and_l2_Modules import lasso_model, ridge_model
from apps.Cloned2.lib.PBS.SKlearn.All_Regression_objects import RandomForest_Regressor, XGBoost, K_Neighbors_Regressor, \
    Multilayer_Perceptron_Regressor, Support_vector_regression
from apps.Cloned2.lib.PBS.Keras_Tensorflow.Deep_N_Nets_Regressor import Deep_Neural_Nets, lstm
import numpy as np
from sklearn.model_selection import train_test_split
from apps.Cloned2.lib.PBS.SKlearn_grids.Default_Grids import RandomForest_Classifier_grids, XGBoost_Classifier_grids, \
    KNeighbors_Classifier_grids, SVC_Classifier_grids, Multilayer_Perceptron_Classifier_grids
from apps.Cloned2.lib.PBS.SKlearn_grids.User_Grids import User_RandomForest_Classifier_grids, \
    User_XGBoost_Classifier_grids, User_KNeighbors_Classifier_grids, User_SVC_Classifier_grids, \
    User_Multilayer_Perceptron_Classifier_grids


class ALL_RGN_Modeling():
    def __init__(self, data_sorted, y, i, types, dname, config_object_user,
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
                 db_name, learning_rate_init, max_iter, hidden_layer_sizes, activation, alpha, early_stopping,
                 dataset_type, test_size, samples):
        self.data_sorted = data_sorted
        self.i = i
        self.types = types
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
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.early_stopping = early_stopping
        self.dataset_type = dataset_type
        self.test_size = test_size
        self.samples = samples
        self.RandomForest_Classifier_grids = RandomForest_Classifier_grids()
        self.XGBoost_Classifier_grids = XGBoost_Classifier_grids()
        self.KNeighbors_Classifier_grids = KNeighbors_Classifier_grids()
        self.SVC_Classifier_grids = SVC_Classifier_grids()
        self.Multilayer_Perceptron_Classifier_grids = Multilayer_Perceptron_Classifier_grids()
        self.RandomForest_Classifier_grids = User_RandomForest_Classifier_grids(self.min_depth, self.max_depth,
                                                                                self.min_samples_split,
                                                                                self.n_estimators_start,
                                                                                self.n_estimators_stop,
                                                                                self.RandomForest_Classifier_grids,
                                                                                self.default)
        self.XGBoost_Classifier_grids = User_XGBoost_Classifier_grids(self.xgb_objective, self.xgb_learning_rate,
                                                                      self.xgb_max_depth, self.xgb_min_child_weight,
                                                                      self.n_estimators_start,
                                                                      self.n_estimators_stop,
                                                                      self.XGBoost_Classifier_grids, self.default)
        self.KNeighbors_Classifier_grids = User_KNeighbors_Classifier_grids(self.n_neighbors,
                                                                            self.KNeighbors_Classifier_grids,
                                                                            self.default)
        self.SVC_Classifier_grids = User_SVC_Classifier_grids(self.svm_c, self.svm_gamma, self.svm_kernel,
                                                              self.SVC_Classifier_grids, self.default)
        self.Multilayer_Perceptron_Classifier_grids = User_Multilayer_Perceptron_Classifier_grids(
            self.learning_rate_init, self.max_iter, self.hidden_layer_sizes, self.activation,
            self.alpha, self.early_stopping, self.Multilayer_Perceptron_Classifier_grids, self.default)

    def Building_Models_Reg(self):
        #         try:
        dname = self.dname
        user_defined_terminology = self.user_defined_terminology
        sample_type = self.sample_type
        description = self.description
        uom_type = self.uom_type
        config_object_user = self.config_object_user
        cut_off = self.cut_off
        analysis_id = self.analysis_id
        db_name = self.db_name
        dataset_type = self.dataset_type
        test_size = self.test_size
        samples = self.samples
        X = self.data_sorted.drop([self.i], axis=1)
        Y = self.data_sorted[self.i]
        X = X.fillna(X.mean())
        y = (', '.join(["%s" % self.i]))
        cols = list(self.data_sorted.columns)
        x = cols
        x.remove(y)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
        X_train, X_test, result = train_test(X_train, X_test, cut_off)
        print(X_test)
        target = self.i
        l = 1
        Samples = X_test[samples]
        print(Samples)
        X_train = X_train.drop(samples, 1)
        X_test = X_test.drop(samples, 1)
        # lasso_model(X_train, y_train, X_test, y_test)
        # ridge_model(X_train, y_train, X_test, y_test)
        def modelGeneratorFun():
            yield RandomForest_Regressor, self.RandomForest_Classifier_grids
            yield XGBoost, self.XGBoost_Classifier_grids
            yield K_Neighbors_Regressor, self.KNeighbors_Classifier_grids
            yield Multilayer_Perceptron_Regressor, self.Multilayer_Perceptron_Classifier_grids
            yield Support_vector_regression, self.SVC_Classifier_grids

        # for model, grids in modelGeneratorFun():
        #     #                 try:
        #     model_ = "Sklearn_Reg_" + str(analysis_id)
        #     train_accuracy, test_accuracy, random_best, importances, grid, estimator, model_file_name, models ,mbe= model(
        #         X_train, y_train, X_test, y_test, grids, target, model_)
        #     user_defined_terminology = self.user_defined_terminology
        #     df = pd.DataFrame({"Sample": Samples, "Target": y_test, "Prediction": random_best })
        #     DB_upload(train_accuracy, test_accuracy, X_train, X_test, y_test, random_best,
        #               importances, grid, estimator, l, "None", target, model_file_name, models, dname,
        #               config_object_user,
        #               user_defined_terminology, sample_type, description, uom_type, analysis_id, db_name, result,
        #               dataset_type, df ,mbe)

        #
        def Keras_Regressor_():
            # yield Deep_Neural_Nets
            yield lstm

        for model in Keras_Regressor_():
            #               try:
            model_ = "tf.Keras_Reg_Deep_Neural_Nets_" + str(analysis_id) + "_"
            train_accuracy, test_accuracy, random_best, importances, grids, estimator, model_file_name, model_name, cm, mbe = model( X_train, y_train, X_test, y_test, target, model_)
            user_defined_terminology = self.user_defined_terminology
            df = pd.DataFrame({"Sample": Samples, "Target": y_test, "Prediction": random_best })
            DB_upload(train_accuracy, test_accuracy, X_train, X_test, y_test, random_best, importances,
                      grids, estimator, l, "None", target, model_file_name, model_name, dname, config_object_user,
                      user_defined_terminology, sample_type, description, uom_type, analysis_id, db_name, result,
                      dataset_type, df, mbe)
        import h2o
        h2o.init()
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
        db_train = X_train.drop(y, axis=1)
        db_test = X_test.drop(y, axis=1)

        def modelGeneratorFun():
            yield h2o_RF
            yield h2o_GB
            # yield h2o_XGB
            yield h2o_Glm
            yield h2o_DeepNN

        for model in modelGeneratorFun():
            #                     try:
            train_accuracy, test_accuracy, cm, mse, rmse, r2, importances, grid, model_file_name, model, variable_order, y_pred, mbe = model(
                train, X_test, x, y, y_test, str(analysis_id))
            db_train = db_train[list(variable_order)]
            db_test = db_test[list(variable_order)]
            user_defined_terminology = self.user_defined_terminology
            df = pd.DataFrame({"Sample": Samples, "Target": y_test, "Prediction": y_pred})
            h2o_DB_upload(train_accuracy, test_accuracy, db_train, db_test, rmse, mse, r2, importances,
                          grid, " ", cm, y, model_file_name, model, dname, config_object_user, user_defined_terminology,
                          sample_type, description, uom_type, analysis_id, db_name, result, dataset_type, df, mbe)

#                     except Model_building_fail:
#                         print("Sorry ! You are dividing by zero ")
