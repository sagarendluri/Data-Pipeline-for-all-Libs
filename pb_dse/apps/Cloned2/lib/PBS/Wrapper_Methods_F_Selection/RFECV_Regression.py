import pandas as pd
import numpy as np
# from statsmodels.stats.outliers_influence import variance_inflation_factor
from apps.Cloned2.lib.PBS.Wrapper_Methods_F_Selection.Wrapper_Methods_Models.Regression_Models import RFECV_RF_Regressor , RFECV_XGB_Regressor
from apps.Cloned2.lib.PBS.Data_preprocessing.QC import train_test
from apps.Cloned2.lib.PBS.Upload_Data_Database.Fill_DB import DB_upload
from sklearn.model_selection import train_test_split
from apps.Cloned2.lib.PBS.SKlearn_grids.Default_Grids import RandomForest_Classifier_grids,XGBoost_Classifier_grids#,KNeighbors_Classifier_grids,SVC_Classifier_grids,Multilayer_Perceptron_Classifier_grids
from apps.Cloned2.lib.PBS.SKlearn_grids.User_Grids import User_RandomForest_Classifier_grids,User_XGBoost_Classifier_grids#,User_KNeighbors_Classifier_grids,User_SVC_Classifier_grids,User_Multilayer_Perceptron_Classifier_grids

import warnings
warnings.filterwarnings('ignore')
class ALL_RGN_Modeling:
    def __init__(self, data_sorted, y, i, types, dname, config_object_user,
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
                 db_name,N_features,learning_rate_init ,max_iter, hidden_layer_sizes, activation,alpha,early_stopping):
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
        self.N_features = N_features
        self.learning_rate_init= learning_rate_init
        self.max_iter= max_iter
        self.hidden_layer_sizes =hidden_layer_sizes
        self.activation =activation
        self.alpha =alpha
        self.early_stopping =early_stopping
        self.RandomForest_Classifier_grids=RandomForest_Classifier_grids()
        self.XGBoost_Classifier_grids=XGBoost_Classifier_grids()
        self.RandomForest_Classifier_grids = User_RandomForest_Classifier_grids(self.min_depth,self.max_depth,self.min_samples_split,self.n_estimators_start,
                                        self.n_estimators_stop,self.RandomForest_Classifier_grids,self.default)
        self.XGBoost_Classifier_grids = User_XGBoost_Classifier_grids(self.xgb_objective,self.xgb_learning_rate,self.xgb_max_depth,self.xgb_min_child_weight,self.n_estimators_start,
                                        self.n_estimators_stop,self.XGBoost_Classifier_grids,self.default)

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
            N_features = self.N_features
            X = self.data_sorted.drop([self.i], axis=1)
            Y = self.data_sorted[self.i]
            X = X.fillna(X.mean())
            y = (', '.join(["%s" % self.i]))
            cols = list(self.data_sorted.columns)
            x = cols
            x.remove(y)
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
            X_train, X_test ,result = train_test(X_train, X_test, cut_off)
            print("before_build",X_train.shape)
            target = self.i
            l=1
            def modelGeneratorFun():
                    yield RFECV_RF_Regressor , self.RandomForest_Classifier_grids
                    yield RFECV_XGB_Regressor , self.XGBoost_Classifier_grids
            for model,grids in modelGeneratorFun():
#                 try:
                    model = "RFECV_Regressor"
                    Accuracy,X_train,X_test,y_test,random_best,importances,grid,estimator,l,cm,target,model_file_name,model= model(X_train , y_train,X_test, y_test,grids, target)
                    user_defined_terminology = self.user_defined_terminology
                    DB_upload(Accuracy,X_train,X_test,y_test,random_best,
                                    importances,grid,estimator,l,"None",target,model_file_name,models,dname,config_object_user,
                                user_defined_terminology,sample_type ,description ,uom_type,analysis_id,db_name,result)
