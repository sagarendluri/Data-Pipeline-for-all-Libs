from scipy import stats
import numpy as np
from apps.Cloned2.lib.PBS.SKlearn.All_Regression_objects import RandomForest_Regressor, XGBoost, K_Neighbors_Regressor, \
    Multilayer_Perceptron_Regressor, Support_vector_regression
from apps.Cloned2.lib.PBS.Keras_Tensorflow.Deep_N_Nets_Regressor import Deep_Neural_Nets
from apps.Cloned2.lib.PBS.Data_preprocessing.QC import train_test
from apps.Cloned2.lib.PBS.Upload_Data_Database.Fill_DB import DB_upload
from apps.Cloned2.lib.PBS.Upload_Data_Database.Fill_h2o_DB import h2o_DB_upload
import apps.Cloned2.lib.PBS.Upload_Data_Database.DB_details
from apps.Cloned2.lib.PBS.H2O.h2o_Reg import h2o_RF, h2o_GB, h2o_XGB, h2o_Glm, h2o_DeepNN
from sklearn.model_selection import train_test_split
from apps.Cloned2.lib.PBS.SKlearn_grids.Default_Grids import RandomForest_Classifier_grids, XGBoost_Classifier_grids, \
    KNeighbors_Classifier_grids, SVC_Classifier_grids, Multilayer_Perceptron_Classifier_grids
from apps.Cloned2.lib.PBS.SKlearn_grids.User_Grids import User_RandomForest_Classifier_grids, \
    User_XGBoost_Classifier_grids, User_KNeighbors_Classifier_grids, User_SVC_Classifier_grids, \
    User_Multilayer_Perceptron_Classifier_grids
def x2(df, col2):
    cols = []
    for col1 in df.columns:
        df_cont = pd.crosstab(index=df[col1], columns=df[col2])
        degree_f = (df_cont.shape[0] - 1) * (df_cont.shape[1] - 1)
        df_cont.loc[:, 'Total'] = df_cont.sum(axis=1)
        df_cont.loc['Total'] = df_cont.sum()
        df_exp = df_cont.copy()
        df_exp.iloc[:, :] = np.multiply.outer(
            df_cont.sum(1).values, df_cont.sum().values) / df_cont.sum().sum()
        df_chi2 = ((df_cont - df_exp) ** 2) / df_exp
        df_chi2.loc[:, 'Total'] = df_chi2.sum(axis=1)
        df_chi2.loc['Total'] = df_chi2.sum()
        chi_square_score = df_chi2.iloc[:-1, :-1].sum().sum()
        P = stats.distributions.chi2.sf(chi_square_score, degree_f)
        if P < 0.05:
            print("categorical variables are correlated.")
            cols.append(col1)
        #     return cols
        else:
            print("categorical variables are not correlated.")
    return cols
cols = x2(df, col2)
def x2_models(cols):
    X = self.df[cols]
    y = y
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

    def modelGeneratorFun():
        yield RandomForest_Regressor, self.RandomForest_Classifier_grids
        yield XGBoost, self.XGBoost_Classifier_grids
        yield K_Neighbors_Regressor, self.KNeighbors_Classifier_grids
        yield Multilayer_Perceptron_Regressor, self.Multilayer_Perceptron_Classifier_grids
        yield Support_vector_regression, self.SVC_Classifier_grids

    for model, grids in modelGeneratorFun():
        #                 try:
        model_ = "Sklearn_Reg_" + str(analysis_id)
        train_accuracy, test_accuracy, random_best, importances, grid, estimator, model_file_name, models = model(
            X_train, y_train, X_test, y_test, grids, target, model_)
        user_defined_terminology = self.user_defined_terminology
        df = pd.DataFrame({"Sample": Samples, "Target": y_test, "Prediction": random_best})
        DB_upload(train_accuracy, test_accuracy, X_train, X_test, y_test, random_best,
                  importances, grid, estimator, l, "None", target, model_file_name, models, dname,
                  config_object_user,
                  user_defined_terminology, sample_type, description, uom_type, analysis_id, db_name, result,
                  dataset_type, df)

    #
    def Keras_Regressor_():
        yield Deep_Neural_Nets

    for model in Keras_Regressor_():
        #               try:
        model_ = "Keras_Reg_Deep_Neural_Nets_" + str(analysis_id) + "_"
        train_accuracy, test_accuracy, random_best, importances, grid, estimator, model_file_name, model_name, cm = model(
            X_train, y_train, X_test, y_test, target, model_)
        user_defined_terminology = self.user_defined_terminology
        df = pd.DataFrame({"Sample": Samples, "Target": y_test, "Prediction": random_best})
        DB_upload(train_accuracy, test_accuracy, X_train, X_test, y_test, random_best, importances,
                  grid, estimator, l, "None", target, model_file_name, model_name, dname, config_object_user,
                  user_defined_terminology, sample_type, description, uom_type, analysis_id, db_name, result,
                  dataset_type, df)
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
        train_accuracy, test_accuracy, cm, mse, rmse, r2, importances, grid, model_file_name, model, variable_order, y_pred = model(
            train, X_test, x, y, y_test, str(analysis_id))
        db_train = db_train[list(variable_order)]
        db_test = db_test[list(variable_order)]
        user_defined_terminology = self.user_defined_terminology
        df = pd.DataFrame({"Sample": Samples, "Target": y_test, "Prediction": y_pred})
        h2o_DB_upload(train_accuracy, test_accuracy, db_train, db_test, rmse, mse, r2, importances,
                      grid, " ", cm, y, model_file_name, model, dname, config_object_user, user_defined_terminology,
                      sample_type, description, uom_type, analysis_id, db_name, result, dataset_type, df)

