import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
# from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, RFE, RFECV
from sklearn.decomposition import PCA
import joblib
from dateutil.parser import parse
from apps.Cloned2.lib.PBS.QC import train_test
from apps.Cloned2.lib.PBS.Fill_DB import DB_upload
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier
import warnings
warnings.filterwarnings('ignore')
class sk_Models:
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
        ):
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
            X_train, X_test = train_test(X_train, X_test, cut_off)
            print("before_build",X_train.shape)
            target = self.i
            models = ['Random_Forest',  'XGB']#'MLP_NN',, 'SVR' 'KNN',
            l = 1
            for classifier, params, model in zip(self.Regressor,self.Regressor_grids,  models):
                model_file_name = model + "_" + target + '.pkl'
                user_defined_terminology = user_defined_terminology
                rfe = RFE(estimator = classifier,n_features_to_select = N_features()).fit(X_train, y_train)
                gd = RandomizedSearchCV(classifier,params,cv = 5, n_jobs=-1,verbose=True,refit = True)
                gd.fit(X_train[list(X_train.columns[rfe.support_])], y_train)
#                 rfe2 = classifier.fit(X_train[list(X_train.columns[gd.support_])], y_train)
                print("top",len(list(X_train.columns[rfe.support_])))
                random_best= gd.best_estimator_.predict(X_test[list(X_train.columns[rfe.support_])])
                errors = abs(random_best - y_test)
                mape = np.mean(100 * (errors / y_test))
                Accuracy = 100 - mape
                print(Accuracy)
                grid =gd.best_params_
                estimator = gd.best_estimator_
                model_file_name = model + "_" + target + ".pkl"
                joblib.dump(estimator, model_file_name)
                importances = gd.best_estimator_.feature_importances_
                feature_list = list(X.columns)
                feature_importance = sorted(zip(importances, feature_list), reverse=True)
                df = pd.DataFrame(feature_importance, columns=['importance', 'feature'])
                importance = list(df['importance'])
                feature = list(df['feature'])
                feature_list = list(X_train.columns)
                for k, v in grid.items():
                    try:
                        if v == int:
                            grid[k] = int(v)
                            # grid['learning_rate'] == float(0.045)
                    except ValueError:
                        if v == float:
                            grid[key] = float(value)
                importances = importances.astype(float)
                feature_importance = sorted(zip(importances, feature_list), reverse=True)
                DB_upload(Accuracy,X_train[list(X_train.columns[rfe.support_])],X_test[list(X_train.columns[rfe.support_])],y_test,random_best,
                            importances,grid,estimator,l,"None",target,model_file_name,model,dname,config_object_user,
                        user_defined_terminology,sample_type ,description ,uom_type,analysis_id,db_name)

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
