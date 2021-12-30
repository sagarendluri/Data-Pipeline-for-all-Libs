import pandas as pd
import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA
import eli5
from dateutil.parser import parse
from sklearn.neural_network import MLPRegressor
from sklearn.utils import class_weight
from apps.Cloned2.lib.PBS.Data_preprocessing.QC import train_test
from apps.Cloned2.lib.PBS.Upload_Data_Database.Fill_DB import DB_upload
from sklearn.model_selection import train_test_split
from apps.Cloned2.lib.PBS.SKlearn.All_Regression_objects import RandomForest_Regressor, XGBoost
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import warnings
from apps.Cloned2.lib.PBS.SKlearn_grids.Default_Grids import RandomForest_Classifier_grids, \
    XGBoost_Classifier_grids  # ,KNeighbors_Classifier_grids,SVC_Classifier_grids,Multilayer_Perceptron_Classifier_grids
from apps.Cloned2.lib.PBS.SKlearn_grids.User_Grids import User_RandomForest_Classifier_grids, \
    User_XGBoost_Classifier_grids  # ,User_KNeighbors_Classifier_grids,User_SVC_Classifier_grids,User_Multilayer_Perceptron_Classifier_grids

warnings.filterwarnings('ignore')


class PCA_Unsupervised:
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
                 db_name, learning_rate_init, max_iter, hidden_layer_sizes, activation, alpha, early_stopping,
                 dataset_type,test_size,samples):
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
        samples = self.samples
        X = self.data_sorted.drop([self.i], axis=1)
        print(X)
        Y = self.data_sorted[self.i]
        X = X.fillna(X.mean())
        y = (', '.join(["%s" % self.i]))
        cols = list(self.data_sorted.columns)
        x = cols
        x.remove(y)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        X_train, X_test, result = train_test(X_train, X_test, cut_off)
        target = self.i
        l = 1
        Samples = X_test[samples]
        print(Samples)
        X_train = X_train.drop(samples, 1)
        X_test = X_test.drop(samples, 1)
        def pca_best_component(X_train):
            pca = PCA()
            pca.fit(X_train)
            a = np.round(pca.explained_variance_ratio_.cumsum(), 10)
            b = [value for value in list(set(sorted(a))) if value != 1][-1]
            print(b)
            return b

        b = pca_best_component(X_train)

        def do_pca(b, X_train, X_test):
            pca = PCA(b)
            pca.fit(X_train)
            pca_train = pca.transform(X_train)
            pca_test = pca.transform(X_test)
            n_pca = pca.components_.shape[0]
            most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pca)]
            only_important_names = [X_train.columns[most_important[i]] for i in range(n_pca)]
            X_train_pca = X_train[set(only_important_names)]
            X_test_pca = X_test[set(only_important_names)]
            return X_train_pca, X_test_pca, pca_train, pca_test

        X_train_pca, X_test_pca, pca_train, pca_test = do_pca(b, X_train, X_test)

        def modelGeneratorFun():
            yield RandomForest_Regressor, self.RandomForest_Classifier_grids
            yield XGBoost, self.XGBoost_Classifier_grids

        for model, grids in modelGeneratorFun():
            #                 try:
            model_ = "PCA"
            train_accuracy, test_accuracy, random_best, importances, grid, estimator, model_file_name, models ,mbe = model(pca_train, y_train,
                                                                                                                     pca_test, y_test,
                                                                                     grids, target, model_)
            df = pd.DataFrame({"Sample": Samples, "Target": y_test, "Prediction": random_best})
            user_defined_terminology = self.user_defined_terminology
            DB_upload(train_accuracy, test_accuracy,X_train_pca, X_test_pca, y_test, random_best, importances, grid, estimator, l, "None", target, model_file_name, models, dname,config_object_user,user_defined_terminology, sample_type, description, uom_type, analysis_id, db_name, result, dataset_type, df, mbe)
#                 except:
#                     print("Sorry ! model GOT failed ")
#                 #                 print(grid)
#                 #                 print("Accuracy:",Accuracy)
#                 #                 print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
#                 #                 print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
#                 #                 print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#                 cm = "None"
#                 target = self.i
#                 if model == 'KNN':
#                     from eli5.sklearn import PermutationImportance
#                     perm = PermutationImportance(gd, random_state=1).fit(X_train, y_train)
#                     importances = perm.feature_importances_
#                     feature_list = list(X_train.columns)
#                     feature_importance = sorted(zip(importances, feature_list),
#                                                 reverse=True)  # create two lists from the previous list of tuples
#                     df = pd.DataFrame(feature_importance, columns=['importance', 'feature'])
#                     for k, v in grid.items():
#                         grid[k] = int(v)
#                         DB_upload(Accuracy,X_train,X_test,y_test,y_pred,
#                                     importances,grid,estimator,l,cm,target,model_file_name,model,dname,config_object_user,
#                                 user_defined_terminology,sample_type ,description ,uom_type,analysis_id,db_name)
#                 elif model == 'MLP_NN':
#                     perm = PermutationImportance(gd, random_state=1).fit(X_train, y_train)
#                     importances = perm.feature_importances_
#                     feature_list = list(X_train.columns)
#                     feature_importance = sorted(zip(importances, feature_list),
#                                                 reverse=True)  # create two lists from the previous list of tuples
#                     df = pd.DataFrame(feature_importance, columns=['importance', 'feature'])
#                     print(grid)
#                     # for k, v in grid.items():
#                     #     grid[k] = int(v)
#                     DB_upload(Accuracy, X_train, X_test, y_test, y_pred,
#                               importances, grid, estimator, l, cm, target, model_file_name, model, dname,
#                               config_object_user,
#                               user_defined_terminology, sample_type, description, uom_type, analysis_id, db_name)
#                 elif model == 'SVR':
#                     importances = gd.best_estimator_.coef_
#                     imp = importances.tolist()
#                     importances = imp[0]
#                     feature_list = list(X_train.columns)
#                     feature_importance = sorted(zip(importances, feature_list), reverse=True)
#                     df = pd.DataFrame(feature_importance, columns=['importance', 'feature'])
#                     DB_upload(Accuracy,X_train,X_test,y_test,y_pred,
#                                     importances,grid,estimator,l,cm,target,model_file_name,model,dname,config_object_user,
#                                 user_defined_terminology,sample_type ,description ,uom_type,analysis_id,db_name)
#                 else:
#
# #                 retrun
#


#         else:
#             import ast
#             self.Regressor_grids = [{
#                 'max_depth': [int(x) for x in np.linspace([ast.literal_eval(x) for x in self.min_depth][0],
#                                                           [ast.literal_eval(x) for x in self.max_depth][0], num=5)],
#                 'max_features': ['auto', 'sqrt'],
#                 'min_samples_split': [ast.literal_eval(x) for x in self.min_samples_split],
#
#                 'n_estimators': [int(x) for x in
#                                  np.linspace(start=[ast.literal_eval(x) for x in self.n_estimators_start][0],
#                                              stop=[ast.literal_eval(x) for x in self.n_estimators_stop][0], num=5)]},
#
#                 {'n_neighbors': np.arange(1, [ast.literal_eval(x) for x in self.n_neighbors][0])},
#                 {
#                     'learning_rate': [ast.literal_eval(x) for x in self.xgb_learning_rate],
#                     'max_depth': [ast.literal_eval(x) for x in self.xgb_max_depth],
#                     'min_child_weight': [ast.literal_eval(x) for x in self.xgb_min_child_weight],
#                     'n_estimators': [int(x) for x in np.linspace(start=
#                                                                  [ast.literal_eval(x) for x in self.n_estimators_start][
#                                                                      0],
#                                                                  stop=
#                                                                  [ast.literal_eval(x) for x in self.n_estimators_stop][
#                                                                      0],
#                                                                  num=5)]},
#                 {'C': [ast.literal_eval(x) for x in self.svm_c],
#                  'gamma': self.svm_gamma,
#                  'kernel': self.svm_kernel}]
#             return self.Regressor_grids
